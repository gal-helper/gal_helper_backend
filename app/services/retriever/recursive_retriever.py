"""
递归检索实现

核心算法：
1. 初始查询 → 获取候选文档 (k=initial_k)
2. 评估结果质量 → 判断是否需要递归
3. 生成子问题 → 对候选结果的细化问题
4. 递归检索 → 针对子问题再次检索 (k=intermediate_k)
5. 结果合并 → 去重并重排序 → 返回 top_n
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.langchain import langchain_manager
from app.core.config import config
from .config import RecursiveRetrieverConfig

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """单个检索结果"""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float = 0.0
    retrieval_depth: int = 1
    retrieval_path: List[str] = field(default_factory=list)
    """检索路径，用于追踪这个结果是如何被检索到的"""


@dataclass
class RecursiveRetrievalReport:
    """递归检索报告"""
    total_results: int
    final_results: int
    recursion_depth_used: int
    execution_time: float
    retrieval_tree: Dict[str, Any]
    merge_info: Dict[str, Any]


class RecursiveRetriever:
    """递归检索器"""
    
    def __init__(
        self,
        config: Optional[RecursiveRetrieverConfig] = None,
        vectorstore=None,
    ):
        self.config = config or RecursiveRetrieverConfig()
        self.vectorstore = vectorstore
        self.logger = logging.getLogger(__name__)
        self._retrieval_cache = {}
        
    def set_vectorstore(self, vectorstore):
        """设置向量数据库"""
        self.vectorstore = vectorstore
    
    async def retrieve(
        self,
        query: str,
        topic: Optional[str] = None,
        return_report: bool = False,
    ) -> Tuple[List[Dict[str, Any]], Optional[RecursiveRetrievalReport]]:
        """
        执行递归检索
        
        Args:
            query: 用户查询
            topic: 主题/表名
            return_report: 是否返回详细报告
            
        Returns:
            (最终结果列表, 报告) 或 (最终结果列表, None)
        """
        start_time = datetime.now()
        
        if not self.config.enable_recursion:
            # 单层检索
            docs = await self._single_retrieve(query, self.config.initial_k, topic)
            results = self._docs_to_results(docs, depth=1)
        else:
            # 递归检索
            results, tree = await self._recursive_retrieve(
                query,
                depth=1,
                parent_query=query,
                topic=topic,
            )
        
        # 最终重排序和去重
        final_results = await self._merge_and_rerank(results)
        final_results = final_results[:self.config.final_k]
        
        # 转换为字典格式
        result_dicts = [
            {
                "content": r.content,
                "metadata": r.metadata,
                "relevance_score": float(r.relevance_score),
                "retrieval_depth": r.retrieval_depth,
                "retrieval_path": r.retrieval_path,
            }
            for r in final_results
        ]
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if return_report:
            report = RecursiveRetrievalReport(
                total_results=len(results),
                final_results=len(final_results),
                recursion_depth_used=self._calculate_max_depth(results),
                execution_time=elapsed,
                retrieval_tree=tree if self.config.enable_recursion else {},
                merge_info={"strategy": self.config.merge_strategy},
            )
            return result_dicts, report
        
        return result_dicts, None
    
    async def _single_retrieve(
        self,
        query: str,
        k: int,
        topic: Optional[str] = None,
    ) -> List[Any]:
        """单层检索"""
        try:
            if topic:
                try:
                    vs = await langchain_manager.async_get_vectorstore_for_table(topic)
                except Exception:
                    vs = self.vectorstore or langchain_manager.get_vectorstore()
            else:
                vs = self.vectorstore or langchain_manager.get_vectorstore()
            
            docs = await vs.asimilarity_search(query, k=k)
            return docs
        except Exception as e:
            self.logger.error(f"单层检索失败: {e}")
            return []
    
    async def _recursive_retrieve(
        self,
        query: str,
        depth: int,
        parent_query: str,
        topic: Optional[str] = None,
    ) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        递归检索核心算法
        
        Returns:
            (所有检索结果, 检索树)
        """
        # 基础情况：达到最大深度
        if depth > self.config.max_recursion_depth:
            return [], {}
        
        # 第1步：检索文档
        k = self.config.initial_k if depth == 1 else self.config.intermediate_k
        docs = await self._single_retrieve(query, k, topic)
        
        if not docs:
            return [], {"depth": depth, "query": query, "results": 0}
        
        # 转换为结果对象
        results = self._docs_to_results(docs, depth=depth, parent_query=parent_query)
        
        # 第2步：评估结果质量
        avg_score = np.mean([r.relevance_score for r in results]) if results else 0
        
        tree = {
            "depth": depth,
            "query": query,
            "results": len(results),
            "avg_score": float(avg_score),
            "children": [],
        }
        
        # 第3步：判断是否需要继续递归
        should_recurse = (
            self.config.enable_recursion
            and depth < self.config.max_recursion_depth
            and avg_score < self.config.min_confidence_score
        )
        
        if should_recurse:
            # 第4步：生成子问题
            sub_questions = await self._generate_sub_questions(
                query,
                results,
                self.config.num_sub_questions,
            )
            
            # 第5步：对每个子问题进行递归检索
            for sub_query in sub_questions:
                sub_results, sub_tree = await self._recursive_retrieve(
                    sub_query,
                    depth=depth + 1,
                    parent_query=query,
                    topic=topic,
                )
                results.extend(sub_results)
                tree["children"].append(sub_tree)
        
        return results, tree
    
    async def _generate_sub_questions(
        self,
        original_query: str,
        current_results: List[RetrievalResult],
        num_questions: int,
    ) -> List[str]:
        """
        基于原始查询和当前结果，生成细化的子问题
        
        策略：
        1. 从原始查询中提取关键词
        2. 结合当前结果中出现的关键信息
        3. 生成补充性的问题
        """
        try:
            # 如果没有 LLM，使用启发式方法
            model = langchain_manager.get_chat_model()
            if not model:
                return self._heuristic_sub_questions(original_query)
            
            # 使用 LLM 生成子问题
            result_text = "\n".join([r.content[:200] for r in current_results[:3]])
            
            prompt = f"""
根据以下原始查询和检索到的部分结果，生成 {num_questions} 个更具体的后续查询问题。
这些问题应该用于深化搜索和获取更多相关信息。

原始查询：{original_query}

当前结果摘要：
{result_text}

请生成 {num_questions} 个问题，每行一个。问题应该：
1. 针对原始查询的不同方面
2. 基于当前结果的内容进行深化
3. 能够找到补充信息

只输出问题列表，不需要其他内容：
"""
            
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            
            response = await model.ainvoke(messages)
            questions = [
                q.strip()
                for q in response.content.split("\n")
                if q.strip() and len(q.strip()) > 5
            ]
            
            return questions[:num_questions]
        
        except Exception as e:
            self.logger.warning(f"LLM 子问题生成失败，使用启发式方法: {e}")
            return self._heuristic_sub_questions(original_query)
    
    def _heuristic_sub_questions(self, original_query: str) -> List[str]:
        """
        启发式子问题生成
        
        策略：组合查询的不同部分
        """
        # 简单的启发式方法
        words = original_query.split()
        
        sub_questions = []
        
        # 子问题 1：添加"详细"、"具体"等修饰词
        if len(words) > 0:
            sub_questions.append(f"{original_query}的具体细节是什么？")
        
        # 子问题 2：反向问题
        if len(words) > 1:
            sub_questions.append(f"{' '.join(words[1:])}")
        
        return sub_questions[:self.config.num_sub_questions]
    
    def _docs_to_results(
        self,
        docs: List[Any],
        depth: int = 1,
        parent_query: str = "",
    ) -> List[RetrievalResult]:
        """将 LangChain 文档转换为 RetrievalResult"""
        results = []
        for i, doc in enumerate(docs):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            
            # 计算相关性得分（如果没有则默认）
            relevance_score = metadata.get('relevance_score', 0.5 + (10 - i) * 0.05)
            relevance_score = min(1.0, max(0.0, relevance_score))
            
            result = RetrievalResult(
                content=content,
                metadata=metadata,
                relevance_score=relevance_score,
                retrieval_depth=depth,
                retrieval_path=[parent_query] if parent_query else [],
            )
            results.append(result)
        
        return results
    
    async def _merge_and_rerank(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        合并来自不同层的结果，去重，并重排序
        """
        if not results:
            return []
        
        # 第1步：去重（基于相似度）
        deduped = self._deduplicate(results)
        
        # 第2步：重排序
        if self.config.enable_reranking and len(deduped) > 1:
            deduped = await self._rerank_results(deduped)
        
        # 第3步：按分数排序
        deduped.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return deduped
    
    def _deduplicate(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """去除重复或高度相似的结果"""
        if len(results) <= 1:
            return results
        
        contents = [r.content for r in results]
        
        try:
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(contents)
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 贪心去重
            kept = []
            used = set()
            
            # 按分数从高到低排序
            sorted_indices = sorted(
                range(len(results)),
                key=lambda i: results[i].relevance_score,
                reverse=True,
            )
            
            for idx in sorted_indices:
                if idx in used:
                    continue
                
                kept.append(results[idx])
                
                # 标记相似的结果为已使用
                for other_idx in range(len(results)):
                    if other_idx != idx and similarity_matrix[idx][other_idx] > self.config.deduplication_threshold:
                        used.add(other_idx)
            
            return kept
        
        except Exception as e:
            self.logger.warning(f"去重失败，返回原始结果: {e}")
            return results
    
    async def _rerank_results(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """重排序结果"""
        if self.config.rerank_method == "cosine":
            return self._rerank_cosine(results)
        elif self.config.rerank_method == "cross_encoder":
            return await self._rerank_cross_encoder(results)
        else:
            return results
    
    def _rerank_cosine(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """使用余弦相似度重排序"""
        if len(results) <= 1:
            return results
        
        try:
            # 组合所有内容
            contents = [r.content for r in results]
            
            # 这里应该有原始查询，但我们用第一个的路径作为参考
            primary_query = results[0].retrieval_path[0] if results[0].retrieval_path else "查询"
            
            # 向量化
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_features=100)
            combined = [primary_query] + contents
            tfidf_matrix = vectorizer.fit_transform(combined)
            
            # 计算相似度
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # 更新分数（加权原有分数）
            for i, sim_score in enumerate(similarities):
                results[i].relevance_score = 0.4 * results[i].relevance_score + 0.6 * sim_score
            
            return results
        
        except Exception as e:
            self.logger.warning(f"余弦重排序失败: {e}")
            return results
    
    async def _rerank_cross_encoder(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """使用 CrossEncoder 重排序"""
        try:
            from sentence_transformers import CrossEncoder
            
            if not hasattr(self, '_cross_encoder'):
                self._cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384')
            
            primary_query = results[0].retrieval_path[0] if results[0].retrieval_path else "查询"
            contents = [r.content for r in results]
            
            # 批量计分
            pairs = [[primary_query, content] for content in contents]
            scores = self._cross_encoder.predict(pairs)
            
            # 更新分数
            for i, score in enumerate(scores):
                normalized_score = 1 / (1 + np.exp(-score))  # sigmoid
                results[i].relevance_score = 0.4 * results[i].relevance_score + 0.6 * normalized_score
            
            return results
        
        except Exception as e:
            self.logger.warning(f"CrossEncoder 重排序失败: {e}")
            return results
    
    def _calculate_max_depth(self, results: List[RetrievalResult]) -> int:
        """计算使用的最大深度"""
        if not results:
            return 1
        return max(r.retrieval_depth for r in results)
