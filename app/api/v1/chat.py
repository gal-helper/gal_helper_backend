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
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from langchain_core.documents import Document

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask")
async def ask_question(
        question: str = Form(...),
        use_rag: bool = Form(True),  # 保留参数，但不使用
        chat_message_service: ChatMessageService = Depends(get_chat_message_service)
):
    """
    兼容旧接口：转发到新版的 /completion 流式接口
    返回格式保持与旧版一致（非流式）
    """
    try:
        session_code = f"temp_{uuid6.uuid7()}"

        # 收集流式响应的完整内容
        full_answer = ""
        sources = []

        async for chunk in chat_message_service.chat(session_code, question):
            # 解析 SSE 格式
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
        logger.error(f"/ask 转发失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        vectorstore: AsyncPGVectorStore = Depends(lambda: langchain_manager.get_vectorstore())
):
    tmp_path = None
    try:
        # 保存上传的文件到临时路径
        content = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # 简单的文本处理，实际应该用 TextProcessor
        text_content = content.decode('utf-8')

        # 简单分块
        chunk_size = 1000
        chunks = []
        for i in range(0, len(text_content), chunk_size):
            chunks.append(text_content[i:i + chunk_size])

        docs = []
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "filename": file.filename,
                        "file_type": file.content_type,
                        "chunk_index": i
                    }
                )
            )

        ids = await vectorstore.aadd_documents(docs)

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
    return StreamingResponse(
        chat_message_service.chat(chatSessionCode, askText),
        media_type="text/event-stream",
    )


@router.get("/history_messages")
async def get_history_messages(
        chatSessionCode: str,
        chat_message_service: ChatMessageService = Depends(get_chat_message_service),
):
    return success_response(
        "获取历史消息成功",
        await chat_message_service.get_history_message(chatSessionCode),
    )