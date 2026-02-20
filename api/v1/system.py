from fastapi import APIRouter, HTTPException
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
        }
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "ai": "operational",
            "database": "operational",
        }
    }


# 暂时注释掉需要 rag_processor 的路由
# @router.get("/stats")
# async def get_statistics():
#     try:
#         return {"message": "Stats endpoint temporarily disabled"}
#     except Exception as e:
#         logger.error(f"Error in /stats endpoint: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#
# @router.get("/history")
# async def get_history(limit: int = 10):
#     try:
#         return {
#             "count": 0,
#             "queries": []
#         }
#     except Exception as e:
#         logger.error(f"Error in /history endpoint: {e}")
#         raise HTTPException(status_code=500, detail=str(e))