import asyncpg
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    AsyncSession,
    create_async_engine,
    AsyncEngine,
)
from sqlalchemy import text
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from app.core.config import config
import logging

logger = logging.getLogger(__name__)


class AsyncDatabaseManager:
    def __init__(self):
        self.async_engine: Optional[AsyncEngine] = None
        self._async_session: Optional[async_sessionmaker[AsyncSession]] = None
        self._initialized = False

    async def init_async_database(self):
        if self._initialized:
            return

        logger.info("Initializing async database...")

        # 直接使用 config.ASYNC_DATABASE_URL
        database_url = config.ASYNC_DATABASE_URL
        logger.info(f"Using database URL: {database_url}")

        # 创建异步引擎
        self.async_engine = create_async_engine(
            database_url,
            echo=config.DEBUG,
            pool_size=config.DB_POOL_MIN_SIZE,
            max_overflow=config.DB_POOL_MAX_SIZE - config.DB_POOL_MIN_SIZE,
            pool_pre_ping=True,
        )

        # 测试连接
        try:
            async with self.async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                await conn.commit()
            logger.info("✅ Database connection successful")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

        # 创建异步会话工厂
        self._async_session = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )

        self._initialized = True
        logger.info("Async database initialized successfully.")

    async def close(self):
        """关闭异步数据库引擎和连接池"""
        if self.async_engine:
            logger.info("Closing async database engine...")
            await self.async_engine.dispose()
            self.async_engine = None
            self._async_session = None
            self._initialized = False
            logger.info("Async database engine disposed.")

    @asynccontextmanager
    async def get_async_db(self) -> AsyncGenerator[AsyncSession, None]:
        """获取数据库会话的上下文管理器"""
        if not self._initialized:
            await self.init_async_database()

        async with self._async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise
            finally:
                await session.close()


# 创建全局唯一实例
async_db_manager = AsyncDatabaseManager()


class LangchainConnectionPool:
    """LangGraph 专用的连接池管理"""

    def __init__(self):
        self._pool: Optional[AsyncConnectionPool] = None
        self._initialized = False

    async def connect(self):
        """初始化连接池"""
        if self._initialized:
            return

        logger.info("Initializing Langchain connection pool...")

        # 直接使用 config.LANGCHAIN_DATABASE_URL
        database_url = config.LANGCHAIN_DATABASE_URL
        logger.info(f"Using Langchain database URL: {database_url}")

        # 创建连接池
        self._pool = AsyncConnectionPool(
            database_url,
            min_size=config.DB_POOL_MIN_SIZE,
            max_size=config.DB_POOL_MAX_SIZE,
            open=False,
            timeout=30,
            max_idle=300,
            kwargs={
                "application_name": "langchain_rag",
                "options": "-c statement_timeout=30000",
            }
        )

        # 手动打开连接池
        await self._pool.open()

        # 验证连接池
        try:
            async with self._pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
            logger.info("✅ Langchain connection pool ready")
        except Exception as e:
            logger.error(f"❌ Langchain connection pool validation failed: {e}")
            await self._pool.close()
            raise

        self._initialized = True
        logger.info("Langchain connection pool initialized successfully.")

    async def disconnect(self):
        """关闭连接池"""
        if self._pool and self._initialized:
            logger.info("Closing Langchain connection pool...")
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("Langchain connection pool closed.")

    def get_pool(self) -> AsyncConnectionPool:
        """获取连接池实例"""
        if not self._pool or not self._initialized:
            raise RuntimeError(
                "Langchain connection pool not initialized. Call connect() first."
            )
        return self._pool


langchain_pool = LangchainConnectionPool()


class DatabaseInitializer:
    """数据库初始化器，处理表创建和索引管理"""

    def __init__(self):
        self._initialized = False

    async def ensure_vector_extension(self, conn):
        """确保 vector 扩展已安装"""
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.debug("Vector extension ensured")

    async def ensure_vector_table(self, conn):
        """确保向量表存在"""
        # 检查表是否存在
        result = await conn.execute(text("""
                                         SELECT EXISTS (SELECT
                                                        FROM information_schema.tables
                                                        WHERE table_name = 'document_embeddings')
                                         """))
        row = result.fetchone()
        table_exists = row[0] if row else False

        if not table_exists:
            logger.info("Creating document_embeddings table...")
            await conn.execute(text(f"""
                CREATE TABLE document_embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    langchain_id TEXT,
                    document_content TEXT,
                    embedding vector({config.VECTOR_DIMENSION}),
                    langchain_metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """))
            logger.info("✅ Table document_embeddings created")
        else:
            logger.debug("Table document_embeddings already exists")

    async def create_vector_index(self):
        """创建向量索引 - 使用 asyncpg 直接连接"""
        if config.SKIP_INDEX_CREATION:
            logger.info("Index creation skipped by configuration")
            return

        conn = None
        try:
            # 直接从连接字符串创建连接，完全绕过连接池
            conn = await asyncpg.connect(config.LANGCHAIN_DATABASE_URL)

            # 检查索引是否存在
            row = await conn.fetchrow("""
                                      SELECT 1
                                      FROM pg_indexes
                                      WHERE indexname = 'document_embeddings_embedding_idx'
                                      """)
            index_exists = row is not None

            if not index_exists and config.AUTO_CREATE_INDEX:
                logger.info("🔨 Creating vector index (this may take a while)...")

                # 设置索引创建超时
                await conn.execute("SET statement_timeout = '5min'")

                # 执行 CONCURRENTLY 创建索引
                await conn.execute(f"""
                    CREATE INDEX CONCURRENTLY document_embeddings_embedding_idx
                        ON document_embeddings
                        USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = {config.VECTOR_INDEX_LISTS})
                """)
                logger.info("✅ Vector index created successfully")
            else:
                logger.info("✅ Vector index already exists")

        except Exception as e:
            logger.error(f"Index creation failed: {e}")
            logger.info("You can create it manually later with:")
            logger.info(
                f"  CREATE INDEX CONCURRENTLY document_embeddings_embedding_idx ON document_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = {config.VECTOR_INDEX_LISTS});")
            # 不抛出异常，让系统继续运行
        finally:
            # 确保连接被关闭
            if conn and not conn.is_closed():
                await conn.close()

    async def initialize(self):
        """执行完整的数据库初始化"""
        if self._initialized:
            return

        logger.info("=" * 60)
        logger.info("Starting database initialization")
        logger.info("=" * 60)

        try:
            # 1. 确保连接池已初始化（用于后续操作）
            await langchain_pool.connect()

            # 2. 在事务内创建表结构
            async with async_db_manager.async_engine.begin() as conn:
                await self.ensure_vector_extension(conn)
                await self.ensure_vector_table(conn)

            logger.info("✅ Database schema initialized")

            # 3. 在事务外创建索引 - 使用独立的 asyncpg 连接
            await self.create_vector_index()

            self._initialized = True
            logger.info("=" * 60)
            logger.info("Database initialization completed")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise


# 创建全局初始化器
db_initializer = DatabaseInitializer()