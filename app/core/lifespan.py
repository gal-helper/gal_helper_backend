from contextlib import asynccontextmanager
from fastapi import FastAPI
import logging
import time

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理器"""
    start_time = time.time()

    # --- 启动阶段 ---
    logger.info("=" * 60)
    logger.info("Starting AI RAG API Server")
    logger.info("=" * 60)

    try:
        # 1. 初始化数据库连接池
        from app.core.db import async_db_manager, langchain_pool

        logger.info("Step 1/3: Initializing database connections...")
        await async_db_manager.init_async_database()
        await langchain_pool.connect()

        # 2. 初始化 Langchain 组件（包括数据库表创建）
        from app.core.langchain import langchain_manager

        logger.info("Step 2/3: Initializing Langchain components...")
        await langchain_manager.initialize()

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"✅ Server started successfully in {elapsed:.2f}s")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ Server startup failed: {e}")
        logger.error("=" * 60)
        raise

    yield  # 应用运行中

    # --- 关闭阶段 ---
    logger.info("=" * 60)
    logger.info("Shutting down server...")
    logger.info("=" * 60)

    from app.core.db import langchain_pool as lp, async_db_manager as adm

    await lp.disconnect()
    await adm.close()

    logger.info("✅ Server shutdown complete")