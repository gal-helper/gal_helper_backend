import datetime as dt
import logging
from typing import List, Dict, Any, Optional

from app.core.dependencies import get_db_pool
import asyncpg
from fastapi import Depends

logger = logging.getLogger(__name__)


class RAGHistoryCRUD:
    def __init__(self, pool: asyncpg.Pool = Depends(get_db_pool)):
        self.pool = pool

    async def save_query_history(
        self,
        question: str,
        answer: str,
        used_docs: Optional[List[str]] = None,
        response_time: float = 0.0,
    ) -> bool:
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                       INSERT INTO query_history
                           (question, answer, used_documents, response_time, asked_at)
                       VALUES ($1, $2, $3, $4, $5)
                       """,
                    question,
                    answer,
                    used_docs or [],
                    response_time,
                    dt.datetime.now(),
                )

                return True
        except Exception as e:
            logger.error(f"Failed to save query history: {e}")
            return False

    async def get_statistics(self) -> Dict[str, Any]:
        try:
            async with self.pool.acquire() as conn:
                stats = {}

                stats["documents"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM documents"
                )
                stats["vectorized_documents"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM documents WHERE content_vector IS NOT NULL"
                )
                stats["queries"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM query_history"
                )

                recent = await conn.fetch("""
                                          SELECT question, answer, asked_at
                                          FROM query_history
                                          ORDER BY asked_at DESC LIMIT 10
                                          """)

                stats["recent_queries"] = [
                    {
                        "question": row["question"][:100] + "..."
                        if len(row["question"]) > 100
                        else row["question"],
                        "answer": row["answer"][:100] + "..."
                        if row["answer"] and len(row["answer"]) > 100
                        else row["answer"],
                        "asked_at": row["asked_at"].isoformat(),
                    }
                    for row in recent
                ]

                file_types = await conn.fetch("""
                                              SELECT file_type, COUNT(*) as count
                                              FROM documents
                                              GROUP BY file_type
                                              """)

                stats["file_types"] = {
                    row["file_type"]: row["count"] for row in file_types
                }

                return stats
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}


# --- 关键：定义注入函数 ---
def get_rag_history_crud(pool: asyncpg.Pool = Depends(get_db_pool)) -> RAGHistoryCRUD:
    return RAGHistoryCRUD(pool)
