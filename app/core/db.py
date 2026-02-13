import asyncpg
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    AsyncSession,
    create_async_engine,
    AsyncEngine,
)
from typing import Optional, AsyncGenerator
from app.core.config import config
import logging

logger = logging.getLogger(__name__)


class AsyncDatabaseManager:
    def __init__(self):
        self.async_engine: Optional[AsyncEngine] = None
        self._async_session: Optional[async_sessionmaker[AsyncSession]] = None

    async def init_async_database(self):
        if self._async_session:
            return

        logger.info("Initializing async database...")

        # 1 创建异步引擎
        self.async_engine = create_async_engine(
            config.ASYNC_DATABASE_URL,
            echo=True,  # 可选：输出SQL日志
            pool_size=10,  # 设置连接池中保持的持久连接数
            max_overflow=10,  # 设置连接池允许创建的额外连接数
        )

        # 2 创建异步会话工厂
        self._async_session = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Async database initialized.")

    async def close(self):
        """关闭异步数据库引擎和连接池"""
        logger.info("Closing all async database...")

        # 1. 关闭 SQLAlchemy 引擎
        if self.async_engine:
            await self.async_engine.dispose()
            self.async_engine = None
            self._async_session = None
            logger.info("Async database engine disposed.")

    # 依赖项，用于获取数据库会话
    async def get_async_db(self):
        if not self._async_session:
            await self.init_async_database()
            # 重新检查，确保初始化成功
            if not self._async_session:
                raise RuntimeError("Failed to initialize async session.")

        async with self._async_session() as session:
            try:
                yield session  # 返回数据库会话给路由处理函数
                await session.commit()  # 无异常，提交事务
            except Exception as e:
                await session.rollback()  # 有异常，回滚事务
                raise e
            finally:
                await session.close()  # 关闭会话


# 创建全局唯一实例
async_db_manager = AsyncDatabaseManager()


class LangchainConnectionPool:
    """LangGraph AsyncPostgresSaver 专用的连接池管理"""

    def __init__(self):
        self._pool: Optional[AsyncConnectionPool] = None

    async def connect(self):
        """初始化连接池"""
        if self._pool:
            return
        self._pool = AsyncConnectionPool(
            config.LANGCHAIN_DATABASE_URL,
            min_size=10,
            max_size=15,
        )
        await self._pool.open()
        logger.info(
            f"Langchain connection pool initialized: {config.DB_HOST}/{config.DB_NAME}"
        )

    async def disconnect(self):
        """关闭连接池"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Langchain connection pool closed.")

    def get_pool(self) -> AsyncConnectionPool:
        """获取连接池实例"""
        if not self._pool:
            raise RuntimeError(
                "Langchain connection pool not initialized. Call connect() first."
            )
        return self._pool


langchain_pool = LangchainConnectionPool()
