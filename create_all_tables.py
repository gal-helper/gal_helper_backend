#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建所有数据库表
包括：
1. ai_documents - 知识库文档（单表模式）
2. ai_chat_session_info - 聊天会话
3. ai_message_info - 聊天消息
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'ai_knowledge_db')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '12345678b')


async def create_tables():
    """创建所有表"""
    
    try:
        import asyncpg
        
        print("🔌 连接数据库...")
        conn = await asyncpg.connect(
            host=DB_HOST,
            port=int(DB_PORT),
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        print("✅ 数据库连接成功")
        
        # 创建 pgvector 扩展
        print("\n📦 创建 pgvector 扩展...")
        try:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            print("✅ pgvector 扩展创建成功")
        except Exception as e:
            print(f"⚠️  pgvector 扩展: {e}")
        
        # ===== 1. 创建 ai_documents 表（知识库） =====
        print("\n📝 创建 ai_documents 表（知识库）...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_documents (
            id SERIAL PRIMARY KEY,
            doc_hash VARCHAR(64) UNIQUE NOT NULL,
            title VARCHAR(512) NOT NULL,
            content TEXT NOT NULL,
            content_type VARCHAR(50),
            source_url VARCHAR(1024),
            embedding vector(1536),
            embedding_model VARCHAR(100) DEFAULT 'nomic-embed-text',
            keywords TEXT[],
            tags JSONB,
            metadata JSONB,
            is_indexed BOOLEAN DEFAULT FALSE,
            is_tagged BOOLEAN DEFAULT FALSE,
            retrieval_count INTEGER DEFAULT 0,
            relevance_score FLOAT DEFAULT 0.0,
            create_time TIMESTAMP DEFAULT NOW(),
            update_time TIMESTAMP DEFAULT NOW()
        );
        """)
        print("✅ ai_documents 表创建成功")
        
        # 创建索引
        try:
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embedding_ivf 
                ON ai_documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            print("✅ 向量索引创建成功")
        except Exception as e:
            print(f"⚠️  向量索引: {e}")
        
        # ===== 2. 创建 ai_chat_session_info 表（会话） =====
        print("\n📝 创建 ai_chat_session_info 表（会话）...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_chat_session_info (
            id SERIAL PRIMARY KEY,
            chat_session_code VARCHAR(100) UNIQUE NOT NULL,
            user_intent INTEGER,
            current_message_id INTEGER,
            create_time TIMESTAMP DEFAULT NOW(),
            update_time TIMESTAMP DEFAULT NOW()
        );
        """)
        print("✅ ai_chat_session_info 表创建成功")
        
        # 创建序列
        try:
            await conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS ai_chat_session_info_id_seq;
            """)
            print("✅ 会话序列创建成功")
        except Exception as e:
            print(f"⚠️  会话序列: {e}")
        
        # ===== 3. 创建 ai_message_info 表（消息） =====
        print("\n📝 创建 ai_message_info 表（消息）...")
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_message_info (
            id SERIAL PRIMARY KEY,
            fk_session_id INTEGER NOT NULL,
            message_id INTEGER NOT NULL,
            parent_id INTEGER,
            role VARCHAR(20) NOT NULL,
            message TEXT,
            create_time TIMESTAMP DEFAULT NOW(),
            update_time TIMESTAMP DEFAULT NOW()
        );
        """)
        print("✅ ai_message_info 表创建成功")
        
        # 创建序列
        try:
            await conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS ai_message_info_id_seq;
            """)
            print("✅ 消息序列创建成功")
        except Exception as e:
            print(f"⚠️  消息序列: {e}")
        
        # 验证表创建
        print("\n✅ 验证表...")
        tables = await conn.fetch("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'ai_%';
        """)
        
        print(f"\n📊 已创建的表 ({len(tables)}):")
        for table in tables:
            print(f"   ✅ {table['table_name']}")
        
        await conn.close()
        
        print("\n" + "="*50)
        print("✅ 所有表创建完成！")
        print("="*50)
        return True
    
    except ImportError:
        print("❌ 缺少 asyncpg 包")
        print("请运行: pip install asyncpg")
        return False
    
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 开始创建所有表...")
    print(f"📊 数据库: {DB_NAME}")
    print()
    
    success = asyncio.run(create_tables())
    
    if success:
        print("\n✅ 表创建成功！可以开始使用系统了。")
    else:
        print("\n❌ 表创建失败，请检查错误信息。")
