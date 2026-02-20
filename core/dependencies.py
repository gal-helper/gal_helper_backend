from sqlalchemy.ext.asyncio import AsyncSession
from app.core.db import async_db_manager

async def get_db() -> AsyncSession:
    """统一数据库会话依赖"""
    async with async_db_manager.get_async_db() as session:
        yield session