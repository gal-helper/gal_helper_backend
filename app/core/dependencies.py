import asyncpg
from typing import AsyncGenerator
from fastapi import Depends
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import db_manager, async_db_manager
from app.core.langchain import langchain_manager

# --- 1. 基础依赖：获取连接池 ---
async def get_db_pool() -> asyncpg.Pool:
    """
    直接返回全局唯一的数据库连接池实例。
    用于需要频繁操作数据库或需要手动控制 acquire() 的场景。
    """
    return db_manager.get_pool()

# --- 2. 进阶依赖：获取单个数据库连接 (自动管理) ---
async def get_db_conn(
    pool: asyncpg.Pool = Depends(get_db_pool)
) -> AsyncGenerator[asyncpg.Connection, None]:
    """
    通过 yield 实现上下文管理。
    FastAPI 会在调用你的路由前执行 yield 之前的代码（获取连接），
    在路由执行完返回响应后，执行 yield 之后的代码（释放连接回池）。
    """
    async with pool.acquire() as connection:
        yield connection

# --- 3. 获取orm管理的session ---
async def get_async_dbsession() -> AsyncGenerator[AsyncSession, None]:
    """
    返回一个异步的orm管理的dbsession
    """
    return async_db_manager.get_async_db()

# --- 4. LLM，获取basemodel ---
async def get_base_model() -> BaseChatModel:
    """
    直接返回全局唯一的basemodel实例，
    用于大模型的调用
    """
    return langchain_manager.get_base_chat_model()