import asyncpg
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, create_async_engine, AsyncEngine
from typing import Optional, AsyncGenerator
from app.core.config import config
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        # 内部保存连接池对象
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """初始化连接池"""
        if self._pool:
            return

        basic_params = {
            'host': config.DB_HOST,
            'port': config.DB_PORT,
            'database': config.DB_NAME,
            'user': config.DB_USER,
            'password': config.DB_PASSWORD
        }
        try:
            # 使用之前在 config.py 封装好的 database_params 字典
            # ** 语法会自动解包为 host, port, user 等参数
            self._pool = asyncpg.create_pool(
                **basic_params,
                min_size=config.DB_POOL_MIN_SIZE,
                max_size=config.DB_POOL_MAX_SIZE,
                command_timeout=config.DB_POOL_TIMEOUT
            )
            logger.info(f"Connected to database: {config.DB_HOST}/{config.DB_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise e

    async def disconnect(self):
        """关闭连接池"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection pool closed.")

    def get_pool(self) -> asyncpg.Pool:
        """获取连接池实例"""
        if not self._pool:
            raise RuntimeError("Database pool is not initialized. Call connect() first.")
        return self._pool

# 创建全局唯一的实例
db_manager = DatabaseManager()

class AsyncDatabaseManager:
    def __init__(self):
        self._async_engine: Optional[AsyncEngine] = None
        self._async_session: Optional[async_sessionmaker[AsyncSession]] = None
        self._raw_pool: Optional[AsyncConnectionPool] = None  # 新增：原始连接池 用于 Langchain

    async def init_async_database(self):
        if self._async_session:
            return

        logger.info("Initializing async database...")

        # 1. 初始化底层的 psycopg 连接池（给 LangGraph 用）
        self._raw_pool = AsyncConnectionPool(conninfo=config.LANGCHAIN_DATABASE_URL, max_size=10, open=False)
        await self._raw_pool.open()

        # 2. 创建异步引擎
        self._async_engine = create_async_engine(
            config.ASYNC_DATABASE_URL,
            echo=True, # 可选：输出SQL日志
            pool_size=10, # 设置连接池中保持的持久连接数
            max_overflow=10, # 设置连接池允许创建的额外连接数
        )

        # 3. 创建异步会话工厂
        self._async_session = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Async database initialized.")

    async def close(self):
        """关闭异步数据库引擎和连接池"""
        logger.info("Closing all async database...")

        # 1. 关闭 SQLAlchemy 引擎
        if self._async_engine:
            await self._async_engine.dispose()
            self._async_engine = None
            self._async_session = None
            logger.info("Async database engine disposed.")
        # 2. 关闭 psycopg 原始连接池 (给 LangGraph 用的那个)
        if self._raw_pool:
            # close 会断开所有空闲连接
            await self._raw_pool.close()
            self._raw_pool = None
            logger.info("LangGraph raw connection pool closed.")

    # 依赖项，用于获取数据库会话
    async def get_async_db(self) -> AsyncGenerator[AsyncSession, None]:
        if not self._async_session:
            await self.init_async_database()
            # 重新检查，确保初始化成功
            if not self._async_session:
                raise RuntimeError("Failed to initialize async session.")

        async with self._async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                raise e
            finally:
                await session.close()

    # 新增：专门给 LangchainManager 用的 getter
    async def get_raw_pool(self) -> AsyncConnectionPool:
        if not self._raw_pool:
            raise RuntimeError("Database not initialized")
        return self._raw_pool

# 创建全局唯一实例
async_db_manager = AsyncDatabaseManager()