import os
import tempfile

from fastapi.responses import StreamingResponse

from app.services.ai.rag_processor import rag_processor
from fastapi import APIRouter, UploadFile, HTTPException, Depends
import logging
from fastapi.params import Form, File

from app.services.chat_info import ChatMessageService, get_chat_message_service
from app.utils.response import success_response

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

    try:
        documents = TextProcessor.process_file_content(tmp_path)
        from langchain_core.documents import Document
        docs = []
        for doc in documents:
            docs.append(
                Document(
                    page_content=doc["content"],
                    metadata={
                        "filename": doc["filename"],
                        "file_type": doc["file_type"],
                        **doc["metadata"]
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
        if os.path.exists(tmp_path):
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
