# -*- coding: utf-8 -*-
"""
混合检索系统 - 支持向量+关键词+标签三维搜索
实现：向量相似度 + BM25关键词 + 标签过滤 的融合检索
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import joinedload

from app.models.document import Document, DocumentTagCache, DocumentEmbeddingIndex

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """单个搜索结果"""
    document_id: int
    title: str
    content: str
    similarity_score: float  # 向量相似度
    keyword_score: float  # 关键词匹配度
    tag_score: float  # 标签匹配度
    combined_score: float  # 综合评分
    tags: Dict[str, Any]
    matched_keywords: List[str]
    matched_tags: List[str]
    relevance_reason: str  # 为什么这个文档相关


class HybridRetriever:
    """
    混合检索引擎
    
    设计：
    1. 向量搜索：找出 Top-K 向量相似文档
    2. 关键词搜索：BM25 或全文搜索关键词匹配文档
    3. 标签过滤：按用户指定的标签进行过滤
    4. 融合重排：综合三个维度的得分，重新排序
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.3,
        tag_weight: float = 0.2,
    ):
        """
        初始化混合检索器
        
        Args:
            db_session: 数据库连接
            vector_weight: 向量相似度权重（0-1）
            keyword_weight: 关键词匹配权重（0-1）
            tag_weight: 标签匹配权重（0-1）
        """
        self.db = db_session
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.tag_weight = tag_weight
        
        # 验证权重和为1
        total_weight = vector_weight + keyword_weight + tag_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"权重和不为1.0: {total_weight}，将进行归一化")
            self.vector_weight = vector_weight / total_weight
            self.keyword_weight = keyword_weight / total_weight
            self.tag_weight = tag_weight / total_weight
    
    async def hybrid_search(
        self,
        query: str,
        embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        max_results: int = 20,
    ) -> List[SearchResult]:
        """
        执行混合搜索
        
        Args:
            query: 搜索查询文本（用于关键词提取）
            embedding: 查询的向量表示
            filters: 过滤条件
                - tags: 标签过滤 {"categories": ["技术"], "domains": ["NLP"]}
                - min_importance: 最低重要性分数
                - difficulty: 难度级别
            top_k: 各个维度各取 Top-K 个结果
            max_results: 最多返回多少个结果
        
        Returns:
            按综合评分排序的搜索结果列表
        """
        
        filters = filters or {}
        results_map = {}  # document_id -> SearchResult
        
        # 1. 向量搜索
        logger.info(f"执行向量搜索（维度：{len(embedding)}）")
        vector_results = await self._vector_search(embedding, top_k)
        
        for doc_id, similarity in vector_results:
            if doc_id not in results_map:
                results_map[doc_id] = {
                    "vector_score": similarity,
                    "keyword_score": 0.0,
                    "tag_score": 0.0,
                }
            else:
                results_map[doc_id]["vector_score"] = similarity
        
        # 2. 关键词搜索
        logger.info(f"执行关键词搜索（查询：{query}）")
        keyword_results = await self._keyword_search(query, top_k)
        
        for doc_id, keyword_score, matched_keywords in keyword_results:
            if doc_id not in results_map:
                results_map[doc_id] = {
                    "vector_score": 0.0,
                    "keyword_score": keyword_score,
                    "tag_score": 0.0,
                    "matched_keywords": matched_keywords,
                }
            else:
                results_map[doc_id]["keyword_score"] = keyword_score
                results_map[doc_id]["matched_keywords"] = matched_keywords
        
        # 3. 标签过滤和打分
        if "tags" in filters and filters["tags"]:
            logger.info(f"执行标签过滤（条件：{filters['tags']}）")
            tag_results = await self._tag_filter_search(filters["tags"], top_k)
            
            for doc_id, tag_score, matched_tags in tag_results:
                if doc_id not in results_map:
                    results_map[doc_id] = {
                        "vector_score": 0.0,
                        "keyword_score": 0.0,
                        "tag_score": tag_score,
                        "matched_tags": matched_tags,
                    }
                else:
                    results_map[doc_id]["tag_score"] = tag_score
                    results_map[doc_id]["matched_tags"] = matched_tags
        
        # 4. 计算综合评分
        final_results = []
        for doc_id, scores in results_map.items():
            combined_score = (
                scores["vector_score"] * self.vector_weight +
                scores["keyword_score"] * self.keyword_weight +
                scores["tag_score"] * self.tag_weight
            )
            
            final_results.append({
                "doc_id": doc_id,
                "scores": scores,
                "combined_score": combined_score,
            })
        
        # 5. 按综合评分排序并获取文档详情
        final_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # 6. 获取文档详情并构建 SearchResult 对象
        search_results = []
        for item in final_results[:max_results]:
            doc_id = item["doc_id"]
            scores = item["scores"]
            combined_score = item["combined_score"]
            
            document = await self._get_document_details(doc_id)
            if document is None:
                continue
            
            result = SearchResult(
                document_id=doc_id,
                title=document.title,
                content=document.content,
                similarity_score=scores["vector_score"],
                keyword_score=scores["keyword_score"],
                tag_score=scores["tag_score"],
                combined_score=combined_score,
                tags=document.tags or {},
                matched_keywords=scores.get("matched_keywords", []),
                matched_tags=scores.get("matched_tags", []),
                relevance_reason=self._generate_relevance_reason(scores),
            )
            search_results.append(result)
        
        logger.info(f"混合搜索完成，返回 {len(search_results)} 个结果")
        return search_results
    
    async def _vector_search(
        self,
        embedding: List[float],
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        使用 pgvector 进行向量相似度搜索
        
        Returns:
            List[（document_id, similarity_score）]
        """
        try:
            # 使用 PostgreSQL 的向量余弦相似度操作符
            query = select(
                Document.id,
                (Document.embedding.cosine_distance(embedding)).label("distance")
            ).where(
                Document.is_indexed == True
            ).order_by("distance").limit(top_k)
            
            result = await self.db.execute(query)
            rows = result.fetchall()
            
            # 将距离转换为相似度评分（距离越小=相似度越高）
            # 相似度 = 1 - 距离（假设距离在0-2之间）
            results = [
                (doc_id, max(0.0, 1.0 - float(distance)))
                for doc_id, distance in rows
            ]
            
            logger.debug(f"向量搜索找到 {len(results)} 个结果")
            return results
        
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float, List[str]]]:
        """
        使用 BM25 或全文搜索进行关键词匹配
        
        Returns:
            List[（document_id, keyword_score, matched_keywords）]
        """
        try:
            # 简单的关键词匹配：检查 query 中的词是否在 keywords 数组中
            # 对于更复杂的 BM25，可以使用 PostgreSQL 的 tsvector
            
            # 提取查询中的关键词（简单分词）
            query_keywords = self._extract_keywords(query)
            
            if not query_keywords:
                logger.debug("查询中没有关键词")
                return []
            
            # 使用 ARRAY 包含检查
            placeholders = [f"'{kw}'" for kw in query_keywords]
            where_clause = text(
                f"keywords && ARRAY[{','.join(placeholders)}]::text[]"
            )
            
            query_obj = select(
                Document.id,
                Document.keywords,
            ).where(
                and_(
                    Document.is_indexed == True,
                    where_clause,
                )
            )
            
            result = await self.db.execute(query_obj)
            rows = result.fetchall()
            
            # 计算关键词匹配评分
            results = []
            for doc_id, doc_keywords in rows:
                if doc_keywords is None:
                    continue
                
                # 匹配的关键词
                matched = [kw for kw in doc_keywords if kw in query_keywords]
                
                # 匹配度评分：匹配关键词数 / 总关键词数
                keyword_score = len(matched) / len(query_keywords) if query_keywords else 0.0
                
                results.append((doc_id, keyword_score, matched))
            
            # 按关键词评分排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            logger.debug(f"关键词搜索找到 {len(results)} 个结果，匹配词：{query_keywords}")
            return results[:top_k]
        
        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []
    
    async def _tag_filter_search(
        self,
        tag_filters: Dict[str, Any],
        top_k: int
    ) -> List[Tuple[int, float, List[str]]]:
        """
        使用标签进行过滤和打分
        
        Args:
            tag_filters: 标签过滤条件
                {
                    "categories": ["技术", "教程"],
                    "domains": ["NLP"],
                    "difficulty": "中级",
                    "min_importance": 0.7,
                }
        
        Returns:
            List[（document_id, tag_score, matched_tags）]
        """
        try:
            # 使用 DocumentTagCache 加快标签查询
            conditions = []
            matched_tag_lists = []
            
            if "categories" in tag_filters:
                categories = tag_filters["categories"]
                # JSONB 包含检查
                for cat in categories:
                    conditions.append(
                        func.jsonb_array_elements_text(DocumentTagCache.categories) == cat
                    )
                matched_tag_lists.append(categories)
            
            if "domains" in tag_filters:
                domains = tag_filters["domains"]
                for domain in domains:
                    conditions.append(
                        func.jsonb_array_elements_text(DocumentTagCache.domains) == domain
                    )
                matched_tag_lists.append(domains)
            
            if "difficulty" in tag_filters:
                difficulty = tag_filters["difficulty"]
                conditions.append(DocumentTagCache.difficulty == difficulty)
                matched_tag_lists.append([difficulty])
            
            if not conditions:
                return []
            
            # 查询匹配的文档
            query = select(
                DocumentTagCache.document_id,
                DocumentTagCache.importance,
                func.array_agg(DocumentTagCache.domains).label("matched_tags")
            ).where(
                or_(*conditions)
            ).order_by(
                DocumentTagCache.importance.desc()
            ).limit(top_k)
            
            result = await self.db.execute(query)
            rows = result.fetchall()
            
            # 构建结果：标签匹配度 = 匹配的标签数 / 查询条件数
            results = []
            total_filter_tags = len(matched_tag_lists)
            
            for doc_id, importance, matched_tags in rows:
                tag_score = (importance or 0.5) * 0.5 + 0.5  # 结合重要性评分
                matched_tags_list = matched_tags if matched_tags else []
                
                results.append((doc_id, tag_score, matched_tags_list))
            
            logger.debug(f"标签过滤找到 {len(results)} 个结果")
            return results
        
        except Exception as e:
            logger.error(f"标签过滤失败: {e}")
            return []
    
    async def _get_document_details(self, doc_id: int) -> Optional[Document]:
        """获取文档完整详情"""
        try:
            query = select(Document).where(Document.id == doc_id)
            result = await self.db.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"获取文档详情失败 {doc_id}: {e}")
            return None
    
    @staticmethod
    def _extract_keywords(text: str, max_keywords: int = 5) -> List[str]:
        """
        从文本中提取关键词（简单分词）
        
        实现：
        - 中文：按字符分割
        - 英文：按空格分割，长度 > 3
        """
        
        if not text:
            return []
        
        # 移除特殊字符
        text = text.replace("，", " ").replace("。", " ").replace("！", " ")
        text = text.replace("？", " ").replace("，", " ").replace("'", "")
        
        # 简单分词
        words = text.split()
        
        # 过滤：
        # - 长度 > 1 的词
        # - 移除停用词
        stopwords = {"的", "是", "在", "和", "了", "有", "不", "我", "他", "你", 
                     "大", "小", "多", "很", "个"}
        
        keywords = [
            w for w in words 
            if len(w) > 1 and w not in stopwords
        ][:max_keywords]
        
        return keywords
    
    @staticmethod
    def _generate_relevance_reason(scores: Dict[str, float]) -> str:
        """根据各维度评分生成相关性原因"""
        reasons = []
        
        if scores["vector_score"] > 0.7:
            reasons.append(f"向量相似度高 ({scores['vector_score']:.2%})")
        
        if scores["keyword_score"] > 0.7:
            reasons.append(f"关键词高度匹配 ({scores['keyword_score']:.2%})")
        
        if scores["tag_score"] > 0.5:
            reasons.append(f"标签相符 ({scores['tag_score']:.2%})")
        
        return "；".join(reasons) if reasons else "多维度匹配"


# 性能优化：缓存常用搜索结果
class RetrieverCache:
    """搜索结果缓存"""
    
    def __init__(self, ttl: int = 3600):  # 1小时缓存
        self.ttl = ttl
        self._cache: Dict[str, Tuple[List[SearchResult], datetime]] = {}
    
    def get(self, query_hash: str) -> Optional[List[SearchResult]]:
        """获取缓存的搜索结果"""
        if query_hash in self._cache:
            results, timestamp = self._cache[query_hash]
            if (datetime.now() - timestamp).total_seconds() < self.ttl:
                return results
            else:
                del self._cache[query_hash]
        return None
    
    def set(self, query_hash: str, results: List[SearchResult]):
        """缓存搜索结果"""
        self._cache[query_hash] = (results, datetime.now())
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
