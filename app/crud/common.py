import logging
from app.core.dependencies import get_db_pool
import asyncpg
from fastapi import Depends

logger = logging.getLogger(__name__)

class CommonCRUD:

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def initialize_tables(self) -> bool:
        try:
            async with self.pool.acquire() as conn:
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    logger.info("pgvector extension created or already exists")
                except Exception as e:
                    logger.warning(f"Cannot create pgvector extension: {e}")
                    return False

                await conn.execute("""
                        CREATE TABLE IF NOT EXISTS documents (
                            id SERIAL PRIMARY KEY,
                            filename VARCHAR(255) NOT NULL,
                            file_type VARCHAR(10),
                            content TEXT NOT NULL,
                            content_vector vector(1536),
                            token_count INTEGER DEFAULT 0,
                            file_metadata JSONB DEFAULT '{}',
                            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

                await conn.execute("""
                        CREATE TABLE IF NOT EXISTS query_history (
                            id SERIAL PRIMARY KEY,
                            question TEXT NOT NULL,
                            answer TEXT,
                            used_documents TEXT[],
                            response_time FLOAT,
                            asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

                try:
                    await conn.execute("""
                                       CREATE INDEX IF NOT EXISTS idx_documents_vector
                                           ON documents USING ivfflat (content_vector vector_cosine_ops);
                                       """)
                    logger.info("Vector index created or already exists")
                except Exception as e:
                    logger.warning(f"Cannot create vector index: {e}")

                await conn.execute("""
                                   CREATE INDEX IF NOT EXISTS idx_documents_filename
                                       ON documents(filename);
                                   """)

                await conn.execute("""
                                   CREATE INDEX IF NOT EXISTS idx_query_asked_at
                                       ON query_history(asked_at DESC);
                                   """)

                logger.info("Database tables initialized successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to initialize tables: {e}")
            return False

# --- 关键：定义注入函数 ---
def get_commons_crud(pool: asyncpg.Pool = Depends(get_db_pool)) -> CommonCRUD:
    return CommonCRUD(pool)