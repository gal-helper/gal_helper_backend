from fastapi import APIRouter, HTTPException
from app.services.ai.rag_processor import rag_processor
import logging
router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def root():
    return {
        "message": "AI RAG API Server",
        "status": "running",
        "endpoints": {
            "/health": "Health check",
            "/ask": "Ask a question (POST)",
            "/upload": "Upload document (POST)",
            "/stats": "Get system statistics",
            "/history": "Get query history"
        }
    }

@router.get("/health")
async def health_check():
    stats = await rag_processor.get_stats()
    return {
        "status": "healthy",
        "services": {
            "ai": "operational",
            "database": "operational" if "error" not in stats else "error",
            "rag_processor": "operational"
        }
    }

@router.get("/stats")
async def get_statistics():
    try:
        stats = await rag_processor.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error in /stats endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history(limit: int = 10):
    try:
        stats = await rag_processor.get_stats()
        history = stats.get("recent_queries", [])
        return {
            "count": len(history),
            "queries": history[:limit]
        }
    except Exception as e:
        logger.error(f"Error in /history endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))