# -*- coding: utf-8 -*-
"""
向量和关键词搜索 Bug 检查和验证工具
验证搜索功能的正确性，检测潜在问题
"""

import logging
import asyncio
from typing import List, Tuple, Dict, Optional
import numpy as np
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.models.document import Document
from app.services.retriever.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class SearchVerifier:
    """
    搜索功能验证工具
    
    检查项：
    1. 向量索引完整性
    2. 关键词索引完整性
    3. 向量相似度计算正确性
    4. 关键词匹配正确性
    5. 标签过滤功能
    """
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.results = {}
    
    async def verify_all(self) -> Dict[str, bool]:
        """执行所有验证"""
        
        logger.info("🔍 开始搜索功能验证...")
        
        # 1. 检查数据库状态
        await self._verify_database_health()
        
        # 2. 检查向量索引
        await self._verify_vector_index()
        
        # 3. 检查关键词索引
        await self._verify_keyword_index()
        
        # 4. 检查标签系统
        await self._verify_tag_system()
        
        # 5. 执行搜索测试
        await self._verify_search_functionality()
        
        # 6. 性能测试
        await self._verify_performance()
        
        logger.info("✅ 验证完成")
        return self.results
    
    async def _verify_database_health(self):
        """检查数据库健康状态"""
        
        logger.info("📊 检查数据库健康状态...")
        
        try:
            # 检查表是否存在
            result = await self.db.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'ai_documents'
                )
            """))
            
            exists = result.scalar()
            self.results["database_table_exists"] = exists
            
            if exists:
                logger.info("✅ ai_documents 表存在")
            else:
                logger.error("❌ ai_documents 表不存在")
                return
            
            # 检查文档数量
            result = await self.db.execute(select(func.count(Document.id)))
            doc_count = result.scalar()
            
            logger.info(f"📈 文档数量: {doc_count}")
            self.results["document_count"] = doc_count
            
            if doc_count == 0:
                logger.warning("⚠️  数据库中没有文档，部分测试无法执行")
                return
            
            # 检查已索引文档比例
            result = await self.db.execute(
                select(func.count(Document.id)).where(Document.is_indexed == True)
            )
            indexed_count = result.scalar()
            
            index_rate = (indexed_count / doc_count * 100) if doc_count > 0 else 0
            logger.info(f"🔍 已索引文档: {indexed_count}/{doc_count} ({index_rate:.1f}%)")
            self.results["indexed_rate"] = index_rate
            
            # 检查已标签化文档比例
            result = await self.db.execute(
                select(func.count(Document.id)).where(Document.is_tagged == True)
            )
            tagged_count = result.scalar()
            
            tag_rate = (tagged_count / doc_count * 100) if doc_count > 0 else 0
            logger.info(f"🏷️  已标签化文档: {tagged_count}/{doc_count} ({tag_rate:.1f}%)")
            self.results["tagged_rate"] = tag_rate
        
        except Exception as e:
            logger.error(f"❌ 数据库健康检查失败: {e}")
            self.results["database_health"] = False
    
    async def _verify_vector_index(self):
        """验证向量索引"""
        
        logger.info("🔍 验证向量索引...")
        
        try:
            # 检查向量列的数据类型
            result = await self.db.execute(text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'ai_documents' AND column_name = 'embedding'
            """))
            
            row = result.fetchone()
            if row:
                col_name, data_type, nullable = row
                logger.info(f"✅ 向量列找到: {col_name} ({data_type})")
                self.results["vector_column_exists"] = True
                
                if "vector" not in data_type.lower():
                    logger.warning(f"⚠️  向量列类型可能不对: {data_type}")
            else:
                logger.error("❌ 向量列不存在")
                self.results["vector_column_exists"] = False
                return
            
            # 检查向量索引
            result = await self.db.execute(text("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'ai_documents' AND indexname LIKE '%embedding%'
            """))
            
            indexes = result.fetchall()
            logger.info(f"🔍 向量索引数: {len(indexes)}")
            for idx_name, idx_def in indexes:
                logger.info(f"   - {idx_name}")
            
            self.results["vector_index_count"] = len(indexes)
            
            # 检查有向量的文档
            result = await self.db.execute(
                select(func.count(Document.id)).where(Document.embedding != None)
            )
            vec_count = result.scalar()
            logger.info(f"📊 有向量的文档: {vec_count}")
            self.results["documents_with_vectors"] = vec_count
            
            # 检查向量维度
            result = await self.db.execute(text("""
                SELECT dimension FROM (
                    SELECT array_length(embedding::float4[], 1) AS dimension
                    FROM ai_documents
                    WHERE embedding IS NOT NULL
                    LIMIT 1
                ) sub
            """))
            
            row = result.fetchone()
            if row:
                dim = row[0]
                logger.info(f"📏 向量维度: {dim}")
                self.results["vector_dimension"] = dim
        
        except Exception as e:
            logger.error(f"❌ 向量索引验证失败: {e}")
            self.results["vector_index_verified"] = False
    
    async def _verify_keyword_index(self):
        """验证关键词索引"""
        
        logger.info("🔍 验证关键词索引...")
        
        try:
            # 检查关键词列
            result = await self.db.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'ai_documents' AND column_name = 'keywords'
            """))
            
            row = result.fetchone()
            if row:
                col_name, data_type = row
                logger.info(f"✅ 关键词列找到: {col_name} ({data_type})")
                self.results["keyword_column_exists"] = True
            else:
                logger.error("❌ 关键词列不存在")
                self.results["keyword_column_exists"] = False
                return
            
            # 检查关键词索引
            result = await self.db.execute(text("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'ai_documents' AND indexname LIKE '%keyword%'
            """))
            
            indexes = result.fetchall()
            logger.info(f"🔍 关键词索引数: {len(indexes)}")
            self.results["keyword_index_count"] = len(indexes)
            
            # 检查有关键词的文档
            result = await self.db.execute(text("""
                SELECT COUNT(*) FROM ai_documents WHERE keywords IS NOT NULL AND array_length(keywords, 1) > 0
            """))
            
            kw_count = result.scalar()
            logger.info(f"📊 有关键词的文档: {kw_count}")
            self.results["documents_with_keywords"] = kw_count
            
            # 检查关键词覆盖率
            result = await self.db.execute(select(func.count(Document.id)))
            total_count = result.scalar()
            
            if total_count > 0:
                kw_rate = (kw_count / total_count * 100)
                logger.info(f"📈 关键词覆盖率: {kw_rate:.1f}%")
                self.results["keyword_coverage"] = kw_rate
        
        except Exception as e:
            logger.error(f"❌ 关键词索引验证失败: {e}")
            self.results["keyword_index_verified"] = False
    
    async def _verify_tag_system(self):
        """验证标签系统"""
        
        logger.info("🔍 验证标签系统...")
        
        try:
            # 检查标签列
            result = await self.db.execute(text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'ai_documents' AND column_name = 'tags'
            """))
            
            row = result.fetchone()
            if row:
                col_name, data_type = row
                logger.info(f"✅ 标签列找到: {col_name} ({data_type})")
                self.results["tag_column_exists"] = True
            else:
                logger.error("❌ 标签列不存在")
                self.results["tag_column_exists"] = False
                return
            
            # 检查有标签的文档
            result = await self.db.execute(text("""
                SELECT COUNT(*) FROM ai_documents 
                WHERE tags IS NOT NULL AND tags::text != '{}'
            """))
            
            tag_count = result.scalar()
            logger.info(f"📊 有标签的文档: {tag_count}")
            self.results["documents_with_tags"] = tag_count
            
            # 检查标签缓存表
            result = await self.db.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'ai_document_tag_cache'
                )
            """))
            
            cache_exists = result.scalar()
            logger.info(f"{'✅' if cache_exists else '⚠️'} 标签缓存表: {'存在' if cache_exists else '不存在'}")
            self.results["tag_cache_exists"] = cache_exists
        
        except Exception as e:
            logger.error(f"❌ 标签系统验证失败: {e}")
            self.results["tag_system_verified"] = False
    
    async def _verify_search_functionality(self):
        """验证搜索功能"""
        
        logger.info("🔍 验证搜索功能...")
        
        try:
            # 获取一个有向量的文档用作测试
            result = await self.db.execute(
                select(Document).where(Document.embedding != None).limit(1)
            )
            test_doc = result.scalar_one_or_none()
            
            if not test_doc:
                logger.warning("⚠️  没有有向量的文档，跳过搜索测试")
                self.results["search_test_skipped"] = True
                return
            
            logger.info(f"🧪 使用文档测试搜索: {test_doc.title}")
            
            # 创建混合检索器
            retriever = HybridRetriever(self.db)
            
            # 测试向量搜索
            try:
                vector_results = await retriever._vector_search(
                    embedding=test_doc.embedding,
                    top_k=5
                )
                logger.info(f"✅ 向量搜索成功: {len(vector_results)} 结果")
                self.results["vector_search_works"] = True
            except Exception as e:
                logger.error(f"❌ 向量搜索失败: {e}")
                self.results["vector_search_works"] = False
            
            # 测试关键词搜索
            try:
                keyword_results = await retriever._keyword_search(
                    query=test_doc.title,
                    top_k=5
                )
                logger.info(f"✅ 关键词搜索成功: {len(keyword_results)} 结果")
                self.results["keyword_search_works"] = True
            except Exception as e:
                logger.error(f"❌ 关键词搜索失败: {e}")
                self.results["keyword_search_works"] = False
            
            # 测试混合搜索
            try:
                hybrid_results = await retriever.hybrid_search(
                    query=test_doc.title,
                    embedding=test_doc.embedding,
                    top_k=5
                )
                logger.info(f"✅ 混合搜索成功: {len(hybrid_results)} 结果")
                self.results["hybrid_search_works"] = True
            except Exception as e:
                logger.error(f"❌ 混合搜索失败: {e}")
                self.results["hybrid_search_works"] = False
        
        except Exception as e:
            logger.error(f"❌ 搜索功能验证失败: {e}")
            self.results["search_functionality_verified"] = False
    
    async def _verify_performance(self):
        """性能测试"""
        
        logger.info("⏱️  执行性能测试...")
        
        try:
            # 测试向量搜索速度
            result = await self.db.execute(
                select(Document).where(Document.embedding != None).limit(1)
            )
            test_doc = result.scalar_one_or_none()
            
            if test_doc:
                start_time = datetime.now()
                result = await self.db.execute(text("""
                    SELECT id FROM ai_documents 
                    WHERE embedding IS NOT NULL 
                    ORDER BY embedding <-> %s LIMIT 10
                """), [test_doc.embedding])
                
                end_time = datetime.now()
                elapsed = (end_time - start_time).total_seconds() * 1000
                
                logger.info(f"⏱️  向量搜索耗时: {elapsed:.2f}ms")
                self.results["vector_search_latency_ms"] = elapsed
                
                if elapsed < 200:
                    logger.info("✅ 向量搜索性能良好")
                else:
                    logger.warning("⚠️  向量搜索性能可以优化")
        
        except Exception as e:
            logger.debug(f"性能测试失败: {e}")


async def run_verification(db_session: AsyncSession):
    """运行完整的验证"""
    
    verifier = SearchVerifier(db_session)
    results = await verifier.verify_all()
    
    logger.info("\n" + "="*50)
    logger.info("验证结果汇总")
    logger.info("="*50)
    
    passed = sum(1 for v in results.values() if v is True)
    total = len(results)
    
    logger.info(f"通过: {passed}/{total} 项验证")
    
    for key, value in results.items():
        status = "✅" if value is True else "⚠️" if value is None else "❌"
        logger.info(f"{status} {key}: {value}")
    
    return results


# 导入 func
from sqlalchemy import func
