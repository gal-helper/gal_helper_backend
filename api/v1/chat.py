import os
import json
import tempfile
import uuid6

from fastapi.responses import StreamingResponse
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from fastapi.params import Form, File
import logging

from app.services.chat_info import ChatMessageService, get_chat_message_service
from app.utils.response import success_response
from app.core.langchain import langchain_manager
from langchain_postgres import PGVector
from langchain_core.documents import Document


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask")
async def ask_question(
        question: str = Form(...),
        use_rag: bool = Form(True),
        chat_message_service: ChatMessageService = Depends(get_chat_message_service),
):
    """问答接口"""
    try:
        session_code = f"temp_{uuid6.uuid7()}"

        full_answer = ""
        sources = []

        async for chunk in chat_message_service.chat(session_code, question):
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if data.get("event") == "message":
                        full_answer += data["data"]["content"]
                    elif data.get("event") == "retrieval":
                        sources.append(data["data"])
                except:
                    pass

        return {
            "success": True,
            "question": question,
            "answer": full_answer,
            "sources": sources,
            "rag_used": True,
            "response_time": 0,
            "error": None
        }

    except Exception as e:
        logger.error(f"/ask 失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
):
    """文档上传接口"""
    tmp_path = None
    try:
        vectorstore = langchain_manager.get_vectorstore()
        if not vectorstore:
            raise HTTPException(status_code=500, detail="向量存储未初始化")
        
        content = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        text_content = content.decode('utf-8')

        # 简单分块
        chunk_size = 1000
        chunks = [text_content[i:i + chunk_size] for i in range(0, len(text_content), chunk_size)]

        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "filename": file.filename,
                    "file_type": file.content_type,
                    "chunk_index": i
                }
            )
            for i, chunk in enumerate(chunks)
        ]

        # 单表存储
        ids = vectorstore.add_documents(docs)

        return {
            "success": True,
            "filename": file.filename,
            "document_ids": ids,
            "chunks": len(ids)
        }

    except Exception as e:
        logger.error(f"上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/completion")
async def completion(
        chatSessionCode: str,
        askText: str,
        chat_message_service: ChatMessageService = Depends(get_chat_message_service),
):
    """流式问答接口"""
    return StreamingResponse(
        chat_message_service.chat(chatSessionCode, askText),
        media_type="text/event-stream",
    )


@router.get("/history_messages")
async def get_history_messages(
        chatSessionCode: str,
        chat_message_service: ChatMessageService = Depends(get_chat_message_service),
):
    """获取历史消息"""
    return success_response(
        "获取历史消息成功",
        await chat_message_service.get_history_message(chatSessionCode),
    )
