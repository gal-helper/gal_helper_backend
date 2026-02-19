"""
递归检索模块

提供多层级的文档检索能力，支持：
- 单层检索：直接向量搜索
- 多层递归检索：基于初始结果生成后续问题
- 自适应检索：根据结果质量自动调整策略
"""

from .recursive_retriever import RecursiveRetriever
from .config import RecursiveRetrieverConfig

__all__ = [
    "RecursiveRetriever",
    "RecursiveRetrieverConfig",
]
