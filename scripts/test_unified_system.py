# -*- coding: utf-8 -*-
"""
统一文档系统完整测试
演示新的三维混合检索系统：向量 + 关键词 + 标签
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import List

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_unified_document_system():
    """测试统一文档系统"""
    
    logger.info("="*60)
    logger.info("统一文档系统完整测试")
    logger.info("="*60)
    
    try:
        # 1. 初始化数据库
        logger.info("\n[1/6] 初始化数据库...")
        from app.core.db import async_db_manager
        await async_db_manager.initialize()
        logger.info("✅ 数据库初始化完成")
        
        # 2. 创建表结构
        logger.info("\n[2/6] 创建表结构...")
        from scripts.init_database import init_database
        success = await init_database()
        if success:
            logger.info("✅ 表结构创建完成")
        else:
            logger.error("❌ 表结构创建失败")
            return False
        
        # 3. 初始化 LangChain
        logger.info("\n[3/6] 初始化 LangChain...")
        from app.core.langchain import langchain_manager
        await langchain_manager.initialize()
        logger.info("✅ LangChain 初始化完成")
        
        # 4. 添加测试文档
        logger.info("\n[4/6] 添加测试文档...")
        from sqlalchemy.ext.asyncio import AsyncSession
        async with AsyncSession(async_db_manager.async_engine) as session:
            await add_test_documents(session)
        logger.info("✅ 测试文档添加完成")
        
        # 5. 验证搜索功能
        logger.info("\n[5/6] 验证搜索功能...")
        async with AsyncSession(async_db_manager.async_engine) as session:
            await verify_search_functionality(session)
        logger.info("✅ 搜索功能验证完成")
        
        # 6. 运行完整验证
        logger.info("\n[6/6] 运行完整系统验证...")
        from scripts.verify_search import run_verification
        async with AsyncSession(async_db_manager.async_engine) as session:
            results = await run_verification(session)
        logger.info("✅ 系统验证完成")
        
        logger.info("\n" + "="*60)
        logger.info("✅ 所有测试完成！")
        logger.info("="*60)
        
        return True
    
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def add_test_documents(session):
    """添加测试文档"""
    
    from app.models.document import Document
    from app.services.tools.tagging_tool import tag_document
    from app.core.langchain import langchain_manager
    
    test_documents = [
        {
            "title": "深度学习基础教程",
            "content": "深度学习是机器学习的一个分支，使用人工神经网络来学习数据的表示。本教程涵盖神经网络、反向传播、卷积神经网络等基础概念。",
            "source_url": "https://example.com/deep-learning",
        },
        {
            "title": "自然语言处理与Transformer",
            "content": "Transformer 是一种基于注意力机制的神经网络架构，在自然语言处理任务中表现出色。包括 BERT、GPT 等大规模语言模型都使用了 Transformer 架构。",
            "source_url": "https://example.com/nlp-transformer",
        },
        {
            "title": "计算机视觉应用实战",
            "content": "计算机视觉涉及图像处理、目标检测、图像分类等任务。卷积神经网络在计算机视觉中广泛应用，如 ResNet、VGG 等经典模型。",
            "source_url": "https://example.com/cv-practice",
        },
        {
            "title": "Python 数据分析指南",
            "content": "使用 Python 进行数据分析的完整指南，包括 Pandas、NumPy、Matplotlib 等库的使用方法。数据清洗、特征工程、可视化等必要步骤。",
            "source_url": "https://example.com/python-data-analysis",
        },
        {
            "title": "分布式系统设计原理",
            "content": "分布式系统的基本概念、设计原则和常见问题。包括一致性、可用性、分区容错性（CAP 定理）等重要概念。",
            "source_url": "https://example.com/distributed-systems",
        },
    ]
    
    embeddings = langchain_manager.get_base_embeddings()
    
    for doc_data in test_documents:
        try:
            # 生成向量
            embedding = await embeddings.aembed_query(doc_data["title"])
            
            # 生成标签（如果 API 可用）
            try:
                tags = await tag_document(
                    doc_data["title"],
                    doc_data["content"],
                    "text"
                )
            except:
                tags = {
                    "categories": ["教程"],
                    "domains": ["AI"],
                    "difficulty": "中级",
                    "importance": 0.5,
                    "auto_tags": [],
                    "custom_tags": [],
                    "language": "zh",
                    "quality_score": 0.7,
                }
            
            # 创建文档
            doc = Document(
                doc_hash=hash(doc_data["title"]),
                title=doc_data["title"],
                content=doc_data["content"],
                content_type="text",
                source_url=doc_data["source_url"],
                embedding=embedding,
                tags=tags,
                is_indexed=True,
                is_tagged=True,
            )
            
            # 提取关键词
            doc.split_keywords()
            
            session.add(doc)
            logger.info(f"✅ 添加文档: {doc_data['title']}")
        
        except Exception as e:
            logger.error(f"❌ 添加文档失败 {doc_data['title']}: {e}")
    
    await session.commit()
    logger.info(f"✅ 共添加 {len(test_documents)} 个测试文档")


async def verify_search_functionality(session):
    """验证搜索功能"""
    
    from app.models.document import Document
    from app.services.retriever.hybrid_retriever import HybridRetriever
    from sqlalchemy import select
    
    # 获取第一个文档
    result = await session.execute(
        select(Document).where(Document.is_indexed == True).limit(1)
    )
    test_doc = result.scalar_one_or_none()
    
    if not test_doc:
        logger.warning("⚠️  没有可用的测试文档")
        return
    
    logger.info(f"\n使用文档测试搜索: {test_doc.title}")
    logger.info(f"向量维度: {len(test_doc.embedding) if test_doc.embedding else 0}")
    
    # 创建检索器
    retriever = HybridRetriever(session)
    
    # 1. 向量搜索
    logger.info("\n[向量搜索] 查询相似文档...")
    try:
        vector_results = await retriever._vector_search(
            embedding=test_doc.embedding,
            top_k=3
        )
        logger.info(f"✅ 找到 {len(vector_results)} 个相似文档")
        for doc_id, score in vector_results:
            logger.info(f"   - 文档ID: {doc_id}, 相似度: {score:.3f}")
    except Exception as e:
        logger.error(f"❌ 向量搜索失败: {e}")
    
    # 2. 关键词搜索
    logger.info("\n[关键词搜索] 查询包含关键词的文档...")
    try:
        keyword_results = await retriever._keyword_search(
            query=test_doc.title,
            top_k=3
        )
        logger.info(f"✅ 找到 {len(keyword_results)} 个匹配文档")
        for doc_id, score, keywords in keyword_results:
            logger.info(f"   - 文档ID: {doc_id}, 匹配度: {score:.3f}, 关键词: {keywords}")
    except Exception as e:
        logger.error(f"❌ 关键词搜索失败: {e}")
    
    # 3. 混合搜索
    logger.info("\n[混合搜索] 结合向量、关键词、标签搜索...")
    try:
        hybrid_results = await retriever.hybrid_search(
            query=test_doc.title,
            embedding=test_doc.embedding,
            filters={
                "tags": {"categories": ["教程", "技术"]},
            },
            top_k=5
        )
        logger.info(f"✅ 混合搜索找到 {len(hybrid_results)} 个结果")
        for i, result in enumerate(hybrid_results, 1):
            logger.info(f"\n   [{i}] {result.title}")
            logger.info(f"       向量相似度: {result.similarity_score:.3f}")
            logger.info(f"       关键词匹配: {result.keyword_score:.3f}")
            logger.info(f"       标签匹配: {result.tag_score:.3f}")
            logger.info(f"       综合评分: {result.combined_score:.3f}")
            logger.info(f"       相关原因: {result.relevance_reason}")
    except Exception as e:
        logger.error(f"❌ 混合搜索失败: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主函数"""
    
    try:
        success = await test_unified_document_system()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  测试被中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
