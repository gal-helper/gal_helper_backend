import asyncpg
from typing import Optional
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