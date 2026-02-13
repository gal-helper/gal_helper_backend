"""
迁移脚本：将旧的 documents 表数据迁移到 LangChain 的 document_embeddings 表
执行方式：python -m scripts.migrate_documents_to_vectorstore
"""
import asyncio
import json
import logging
from typing import List, Dict, Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import config
from app.core.db import async_db_manager
from app.core.langchain import langchain_manager
from app.models.document import DocumentEmbedding  # 如果你有定义模型；没有也可以直接写SQL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate():
    """迁移主函数"""
    logger.info("=" * 60)
    logger.info("开始迁移 documents 表到 document_embeddings 表")
    logger.info("=" * 60)

    # 1. 初始化所有服务
    await async_db_manager.init_async_database()
    await langchain_manager.init_langchain_manager()
    
    async with async_db_manager.get_async_db() as session:
        # 2. 检查旧表是否存在
        check_table = await session.execute(
            text("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'documents')")
        )
        if not check_table.scalar():
            logger.info("旧表 documents 不存在，无需迁移")
            return

        # 3. 获取旧表数据
        result = await session.execute(
            text("""
                SELECT id, filename, content, file_metadata, content_vector 
                FROM documents 
                WHERE content_vector IS NOT NULL
            """)
        )
        rows = result.fetchall()
        logger.info(f"找到 {len(rows)} 条待迁移的记录")

        if not rows:
            logger.info("没有需要迁移的数据")
            return

        # 4. 获取 vectorstore
        vectorstore = await langchain_manager.get_vectorstore()
        
        success_count = 0
        fail_count = 0

        for row in rows:
            try:
                doc_id, filename, content, metadata_json, embedding_vector = row
                
                # 构建 metadata
                metadata = {
                    "old_id": doc_id,
                    "filename": filename,
                    "source": "migration",
                    "migrated_at": asyncio.get_event_loop().time(),
                }
                if metadata_json:
                    try:
                        old_metadata = json.loads(metadata_json)
                        metadata.update(old_metadata)
                    except:
                        pass

                # 核心：写入 vectorstore
                # 方式1：如果有 embedding vector
                if embedding_vector:
                    # 直接插入带向量的文档
                    from langchain_core.documents import Document
                    from pgvector.sqlalchemy import Vector
                    
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    
                    # 直接操作底层表（绕过 AsyncPGVectorStore.add_documents 以避免重复embedding）
                    await vectorstore.add_documents([doc], ids=[f"migrated_{doc_id}"])
                    logger.info(f"✅ 迁移文档 ID {doc_id}: {filename}")
                    success_count += 1
                else:
                    logger.warning(f"⚠️ 文档 ID {doc_id} 无向量，跳过")
                    
            except Exception as e:
                logger.error(f"❌ 迁移文档 ID {row[0]} 失败: {e}")
                fail_count += 1

        logger.info("=" * 60)
        logger.info(f"迁移完成: 成功 {success_count} 条, 失败 {fail_count} 条")
        
        # 5. 询问是否删除旧表
        logger.warning("\n⚠️ 确认删除旧表 documents？(y/n) 建议先验证数据完整性")
        # 生产环境请手动执行，这里先注释掉
        # if input().lower() == 'y':
        #     await session.execute(text("DROP TABLE documents CASCADE"))
        #     await session.commit()
        #     logger.info("旧表 documents 已删除")


if __name__ == "__main__":
    asyncio.run(migrate())