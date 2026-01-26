import os
import tempfile
from app.services.ai.rag_processor import rag_processor
from fastapi import APIRouter, UploadFile, HTTPException
import logging
from fastapi.params import Form, File

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask")
async def ask_question(
        question: str = Form(...),
        use_rag: bool = Form(True)
):
    try:
        logger.info(f"Processing question: {question[:100]}...")

        result = await rag_processor.ask_question(question, use_rag)

        response = {
            "success": result["success"],
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "rag_used": use_rag,
            "response_time": result.get("response_time", 0)
        }

        if result.get("error"):
            response["error"] = result["error"]

        return response
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        logger.info(f"Uploading file: {file.filename}")

        result = await rag_processor.process_document(tmp_path)

        os.unlink(tmp_path)

        response = {
            "filename": file.filename,
            "success": result["success"],
            "chunks": result.get("chunks", 0),
            "documents_processed": result.get("documents_processed", 0),
            "message": result.get("message", "")
        }

        if result.get("document_ids"):
            response["document_ids"] = result["document_ids"]

        return response
    except Exception as e:
        logger.error(f"Error in /upload endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))