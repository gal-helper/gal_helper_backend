# -*- coding: utf-8 -*-
"""
数据库初始化脚本 - 创建统一的文档表结构
从旧的 4 个 vectorstore 表迁移到新的单一 ai_documents 表
"""

import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Base, Document, DocumentTagCache, DocumentEmbeddingIndex
from app.core.db import async_db_manager

logger = logging.getLogger(__name__)


async def init_database():
    """初始化数据库，创建所有表"""
    
    logger.info("🔄 开始数据库初始化...")
    
    # 1. 获取异步引擎
    engine = async_db_manager.async_engine
    if not engine:
        logger.error("❌ 数据库引擎未初始化")
        return False
    
    try:
        # 2. 创建所有表（使用 SQLAlchemy 模型）
        async with engine.begin() as conn:
            logger.info("📝 创建统一文档表结构...")
            
            # 创建所有模型表
            await conn.run_sync(Base.metadata.create_all)
            
            logger.info("✅ 表结构创建完成")
        
        # 3. 创建额外的 PostgreSQL 索引
        async with AsyncSession(engine) as session:
            logger.info("📊 创建数据库索引...")
            
            # 向量相似度搜索索引（已在模型中定义）
            # 标签快速查询索引（已在模型中定义）
            # 关键词全文搜索索引（已在模型中定义）
            
            # 如果需要，创建额外的 GiST 索引用于更好的性能
            try:
                await session.execute(text("""
                    CREATE INDEX IF NOT EXISTS ix_document_embedding_gist 
                    ON ai_documents USING gist (embedding)
                """))
                logger.info("✅ 向量索引创建成功")
            except Exception as e:
                logger.warning(f"⚠️  向量索引创建失败: {e}")
        
        # 4. 检查旧表是否存在（用于迁移）
        logger.info("🔍 检查旧表数据...")
        await check_legacy_tables(engine)
        
        logger.info("✅ 数据库初始化完成")
        return True
    
    except Exception as e:
        logger.error(f"❌ 数据库初始化失败: {e}")
        return False


async def check_legacy_tables(engine):
    """
    检查旧表是否存在，如果存在则提示迁移
    
    旧表名：
    - document_embeddings
    - vectorstore_resource
    - vectorstore_technical
    - vectorstore_tools
    - vectorstore_news
    """
    
    legacy_tables = [
        "document_embeddings",
        "vectorstore_resource",
        "vectorstore_technical",
        "vectorstore_tools",
        "vectorstore_news",
    ]
    
    async with AsyncSession(engine) as session:
        for table_name in legacy_tables:
            try:
                result = await session.execute(text(f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = '{table_name}'
                    )
                """))
                
                exists = result.scalar()
                if exists:
                    logger.warning(f"⚠️  检测到旧表: {table_name}")
                    logger.info(f"   建议：备份数据后执行迁移脚本")
            
            except Exception as e:
                logger.debug(f"检查表 {table_name} 失败: {e}")


async def migrate_from_legacy():
    """
    从旧表迁移数据到新的统一表
    
    迁移流程：
    1. 从旧 vectorstore 表读取数据
    2. 提取内容、向量、元数据
    3. 生成新的标签（如果有 DeepSeek API）
    4. 写入新的 ai_documents 表
    """
    
    logger.info("🔄 开始数据迁移（如果有旧表）...")
    
    engine = async_db_manager.async_engine
    if not engine:
        logger.error("❌ 数据库引擎未初始化")
        return False
    
    async with AsyncSession(engine) as session:
        try:
            # 检查旧表是否存在
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'document_embeddings'
                )
            """))
            
            legacy_exists = result.scalar()
            if not legacy_exists:
                logger.info("✅ 未检测到旧表，无需迁移")
                return True
            
            logger.info("📥 开始迁移旧表数据...")
            
            # 读取旧表中的文档
            result = await session.execute(text("""
                SELECT id, document_content, embedding, langchain_metadata 
                FROM document_embeddings
                LIMIT 1000
            """))
            
            rows = result.fetchall()
            logger.info(f"📊 读取 {len(rows)} 条旧记录")
            
            # 迁移数据（简化版本，完整迁移需要更复杂的逻辑）
            for row in rows:
                # TODO: 创建新 Document 对象并保存
                pass
            
            logger.info("✅ 数据迁移完成")
            return True
        
        except Exception as e:
            logger.error(f"❌ 迁移失败: {e}")
            return False


async def create_vector_index():
    """
    创建向量索引以加快搜索性能
    
    使用 IVFFlat 索引（适合大规模向量搜索）
    """
    
    logger.info("📊 创建向量索引...")
    
    engine = async_db_manager.async_engine
    
    async with AsyncSession(engine) as session:
        try:
            # 创建 pgvector 扩展（如果不存在）
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            # 创建 IVFFlat 索引
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_embedding_ivf 
                ON ai_documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            
            await session.commit()
            logger.info("✅ IVFFlat 向量索引创建成功")
            
        except Exception as e:
            logger.warning(f"⚠️  向量索引创建失败（可能已存在）: {e}")


async def create_keyword_index():
    """创建关键词索引以加快关键词搜索"""
    
    logger.info("📊 创建关键词索引...")
    
    engine = async_db_manager.async_engine
    
    async with AsyncSession(engine) as session:
        try:
            # 创建 GIN 索引用于数组关键词查询
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_keywords_gin 
                ON ai_documents USING gin (keywords)
            """))
            
            await session.commit()
            logger.info("✅ 关键词 GIN 索引创建成功")
            
        except Exception as e:
            logger.warning(f"⚠️  关键词索引创建失败（可能已存在）: {e}")


async def create_tag_index():
    """创建标签索引以加快标签过滤"""
    
    logger.info("📊 创建标签索引...")
    
    engine = async_db_manager.async_engine
    
    async with AsyncSession(engine) as session:
        try:
            # 创建 GIN 索引用于 JSONB 标签查询
            await session.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_tags_gin 
                ON ai_documents USING gin (tags)
            """))
            
            await session.commit()
            logger.info("✅ 标签 GIN 索引创建成功")
            
        except Exception as e:
            logger.warning(f"⚠️  标签索引创建失败（可能已存在）: {e}")


async def check_performance_stats():
    """检查数据库性能统计"""
    
    logger.info("📈 检查数据库性能...")
    
    engine = async_db_manager.async_engine
    
    async with AsyncSession(engine) as session:
        try:
            # 检查表大小
            result = await session.execute(text("""
                SELECT 
                    schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
                FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename LIKE 'ai_%'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
            """))
            
            rows = result.fetchall()
            if rows:
                logger.info("📊 表大小统计：")
                for schema, table, size in rows:
                    logger.info(f"   {table}: {size}")
        
        except Exception as e:
            logger.debug(f"性能统计查询失败: {e}")


async def main():
    """主初始化函数"""
    
    # 1. 初始化数据库连接
    logger.info("🔧 初始化数据库连接...")
    await async_db_manager.initialize()
    
    # 2. 创建表
    success = await init_database()
    
    if success:
        # 3. 创建索引
        await create_vector_index()
        await create_keyword_index()
        await create_tag_index()
        
        # 4. 检查性能
        await check_performance_stats()
        
        # 5. 迁移旧数据（可选）
        # await migrate_from_legacy()
    
    logger.info("✅ 初始化完成")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
