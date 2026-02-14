# test_db_connection.py
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import config


async def test_db_connection():
    """测试数据库连接"""
    print("=" * 60)
    print("Testing Database Connection")
    print("=" * 60)

    print(f"Database URL: {config.ASYNC_DATABASE_URL}")

    try:
        # 创建引擎
        engine = create_async_engine(
            config.ASYNC_DATABASE_URL,
            echo=True
        )

        # 测试连接
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            value = result.scalar()
            print(f"✅ Connection successful! SELECT 1 returned: {value}")
            await conn.commit()

        await engine.dispose()
        print("✅ Test completed successfully")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\nPlease check:")
        print("1. PostgreSQL is running")
        print("2. Database exists: CREATE DATABASE ai_knowledge_db;")
        print("3. pgvector extension is installed: CREATE EXTENSION vector;")
        print("4. Connection credentials are correct")


if __name__ == "__main__":
    asyncio.run(test_db_connection())