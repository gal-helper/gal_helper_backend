import logging
from typing import Dict, Optional, Any, List

from app.core.dependencies import get_db_pool
import asyncpg
from fastapi import Depends
import json
import datetime as dt
from app.core.config import config
from app.services.ai.token import count_tokens

logger = logging.getLogger(__name__)


class DocumentsCRUD:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def save_document(
        self,
        filename: str,
        content: str,
        file_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        try:
            token_count = count_tokens(content)

            async with self.pool.acquire() as conn:
                existing = await conn.fetchrow(
                    "SELECT id FROM documents WHERE filename = $1", filename
                )

                if existing:
                    logger.info(f"Document already exists: {filename}")
                    return existing["id"]

                doc_result = await conn.fetchrow(
                    """
                     INSERT INTO documents (filename, file_type, content, token_count, file_metadata, uploaded_at)
                     VALUES ($1, $2, $3, $4, $5, $6) RETURNING id
                     """,
                    filename,
                    file_type,
                    content,
                    token_count,
                    json.dumps(metadata or {}),
                    dt.datetime.now(),
                )

                doc_id = doc_result["id"]
                logger.info(
                    f"Document saved: {filename} (ID: {doc_id}, Tokens: {token_count})"
                )
                return doc_id
        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            return None

    async def update_document_embedding(
        self, doc_id: int, embedding: List[float]
    ) -> bool:
        try:
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                                   UPDATE documents
                                   SET content_vector = $1::vector
                                   WHERE id = $2
                                   """,
                    embedding_str,
                    doc_id,
                )

                return True
        except Exception as e:
            logger.error(f"Failed to update embedding: {e}")
            return False

    async def search_similar_documents(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[Dict[str, Any]]:
        try:
            query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            async with self.pool.acquire() as conn:
                results = await conn.fetch(
                    f"""
                    SELECT 
                        id,
                        filename,
                        content,
                        file_metadata,
                        1 - (content_vector <=> $1::vector) as similarity
                    FROM documents
                    WHERE content_vector IS NOT NULL
                    AND 1 - (content_vector <=> $1::vector) >= $2
                    ORDER BY content_vector <=> $1::vector
                    LIMIT $3
                """,
                    query_embedding_str,
                    config.SIMILARITY_THRESHOLD,
                    limit,
                )

                documents = []
                for row in results:
                    metadata = (
                        json.loads(row["file_metadata"]) if row["file_metadata"] else {}
                    )

                    documents.append(
                        {
                            "id": row["id"],
                            "filename": row["filename"],
                            "content": row["content"],
                            "metadata": metadata,
                            "similarity": float(row["similarity"]),
                        }
                    )

                logger.info(
                    f"Vector search found {len(documents)} documents (threshold: {config.SIMILARITY_THRESHOLD})"
                )
                return documents

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    async def keyword_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        try:
            async with self.pool.acquire() as conn:
                results = await conn.fetch(
                    f"""
                    SELECT 
                        id,
                        filename,
                        content,
                        file_metadata
                    FROM documents
                    WHERE content ILIKE '%' || $1 || '%'
                    LIMIT $2
                """,
                    query,
                    limit,
                )

                documents = []
                for row in results:
                    metadata = (
                        json.loads(row["file_metadata"]) if row["file_metadata"] else {}
                    )
                    documents.append(
                        {
                            "id": row["id"],
                            "filename": row["filename"],
                            "content": row["content"],
                            "metadata": metadata,
                            "similarity": 0.5,
                        }
                    )

                return documents
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []


# --- 关键：定义注入函数 ---
def get_documents_crud(pool: asyncpg.Pool = Depends(get_db_pool)) -> DocumentsCRUD:
    return DocumentsCRUD(pool)
