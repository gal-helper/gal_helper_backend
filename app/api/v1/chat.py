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
async def ask_question(question: str = Form(...), use_rag: bool = Form(True)):
    try:
        logger.info(f"Processing question: {question[:100]}...")

        result = await rag_processor.ask_question(question, use_rag)

        return {
            "success": result["success"],
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "rag_used": use_rag,
            "response_time": result.get("response_time", 0),
            "error": result.get("error"),
        }
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        filename = file.filename or ""
        suffix = os.path.splitext(filename)[1] if filename else ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"Uploading file: {file.filename}")

        result = await rag_processor.process_document(tmp_path)

        os.unlink(tmp_path)

        return {
            "filename": file.filename,
            "success": result["success"],
            "chunks": result.get("chunks", 0),
            "documents_processed": result.get("documents_processed", 0),
            "message": result.get("message", ""),
            "document_ids": result.get("document_ids", []),
        }
    except Exception as e:
        logger.error(f"Error in /upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
