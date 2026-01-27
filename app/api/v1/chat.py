import os
import tempfile
from app.services.ai.rag_processor import rag_processor
from fastapi import APIRouter, UploadFile, HTTPException
import logging
from fastapi.params import Form, File

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
