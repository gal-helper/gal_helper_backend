from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 【启动初始化阶段】 ---
    logger.info("Starting AI RAG API server...")

    # 数据库连接池初始化
    from app.core.db import db_manager, async_db_manager, langchain_pool

    await db_manager.connect()
    await async_db_manager.init_async_database()
    await langchain_pool.connect()

    # 大模型和langchain初始化
    from app.services.ai.rag_processor import rag_processor
    from app.core.langchain import langchain_manager

    await langchain_manager.init_langchain_manager()
    await rag_processor.initialize()

    logger.info("Services initialized successfully")

    yield  # 这里是应用运行的时间

    # --- 【关闭释放阶段】 ---
    from app.core.db import langchain_pool as lp

    await lp.disconnect()
    await db_manager.disconnect()
    await async_db_manager.close()
