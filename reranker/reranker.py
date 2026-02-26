# app/services/ai/reranker.py
"""
重排序模块 - 可选依赖，不影响原有功能
"""
import logging
from typing import Optional, List, Tuple, Any
import importlib.util

logger = logging.getLogger(__name__)


class Reranker:
    """重排序器 - 如果依赖没安装，自动降级"""

    def __init__(self):
        self._model = None
        self._model_name = None
        self._available = None
        self._embedding_model = None
        self._embedding_available = None

    def is_available(self) -> bool:
        """检查重排序是否可用"""
        if self._available is None:
            self._available = importlib.util.find_spec("sentence_transformers") is not None
        return self._available

    def is_embedding_available(self) -> bool:
        """检查向量模型是否可用"""
        if self._embedding_available is None:
            self._embedding_available = importlib.util.find_spec("sentence_transformers") is not None
        return self._embedding_available

    def load_model(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """延迟加载模型"""
        if not self.is_available():
            return None

        if self._model is not None and self._model_name == model_name:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading reranker model: {model_name}")
            self._model = CrossEncoder(model_name, max_length=512)
            self._model_name = model_name
            return self._model
        except Exception as e:
            logger.warning(f"Failed to load reranker model: {e}")
            self._available = False
            return None

    def load_embedding_model(self):
        """延迟加载 embedding 模型，用于向量余弦相似度"""
        if not self.is_embedding_available():
            return None

        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer for vector-based reranking")
            # 使用轻量级模型以提高速度
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            return self._embedding_model
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self._embedding_available = False
            return None

    def rerank(self, query: str, documents: List[str], top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询语句
            documents: 文档列表
            top_k: 返回前k个，None返回全部

        Returns:
            排序后的 (文档内容, 得分) 列表
        """
        if not documents:
            return []

        model = self.load_model()
        if model is None:
            # 降级：返回原始顺序
            return [(doc, 0.0) for doc in documents[:top_k]] if top_k else [(doc, 0.0) for doc in documents]

        try:
            pairs = [[query, doc] for doc in documents]
            scores = model.predict(pairs)

            scored = list(zip(documents, scores))
            scored.sort(key=lambda x: x[1], reverse=True)

            if top_k:
                scored = scored[:top_k]

            return scored
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]] if top_k else [(doc, 0.0) for doc in documents]

    def rerank_by_vector_cosine(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        基于向量余弦相似度的重排序（精准但不依赖 CrossEncoder）

        Args:
            query: 查询语句
            documents: 文档列表
            top_k: 返回前k个，None返回全部

        Returns:
            排序后的 (文档内容, 得分) 列表，得分为 [0, 1] 的余弦相似度
        """
        if not documents:
            return []

        model = self.load_embedding_model()
        if model is None:
            logger.warning("Embedding model not available, returning original order")
            return [(doc, 0.0) for doc in documents[:top_k]] if top_k else [(doc, 0.0) for doc in documents]

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            query_embedding = model.encode([query], convert_to_numpy=True)
            doc_embeddings = model.encode(documents, convert_to_numpy=True)

            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            normalized_scores = (similarities + 1) / 2

            scored = list(zip(documents, normalized_scores.tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)

            if top_k:
                scored = scored[:top_k]

            return scored
        except Exception as e:
            logger.error(f"Vector cosine reranking failed: {e}")
            return [(doc, 0.0) for doc in documents[:top_k]] if top_k else [(doc, 0.0) for doc in documents]


# 全局单例
reranker = Reranker()