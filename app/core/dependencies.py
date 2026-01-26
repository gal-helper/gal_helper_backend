import asyncpg
from typing import AsyncGenerator
from fastapi import Depends
from app.core.db import db_manager

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

# --- 3. (可选) 业务依赖示例：获取用户身份 ---
# 如果你之后要做用户系统，也会写在这里
# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     ...