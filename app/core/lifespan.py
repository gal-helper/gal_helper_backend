from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 【启动初始化阶段】 ---
    logger.info("Starting AI RAG API server...")
    # 数据库连接池初始化
    from app.core.db import db_manager
    await db_manager.connect()

    from app.core.db import async_db_manager
    await async_db_manager.init_async_database()
    # RAG处理服务初始化（使用全局单例）
    from app.services.ai.rag_processor import rag_processor
    # langchain初始化
    from app.core.langchain import langchain_manager
    langchain_manager.init_langchain()

    await rag_processor.initialize()

    logger.info("Services initialized successfully")

    yield  # 这里是应用运行的时间

    # --- 【关闭释放阶段】 ---
    await db_manager.disconnect()
    await async_db_manager.close()
