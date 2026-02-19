"""
递归检索配置
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RecursiveRetrieverConfig:
    """递归检索配置"""
    
    # 基础参数
    enable_recursion: bool = True
    max_recursion_depth: int = 3
    """最大递归深度（1=单层，2=两层，3=三层）"""
    
    initial_k: int = 10
    """初始检索返回的文档数"""
    
    intermediate_k: int = 5
    """中间层返回的文档数"""
    
    final_k: int = 5
    """最终返回的文档数"""
    
    # 递归触发条件
    min_confidence_score: float = 0.6
    """最小置信度分数（0-1），低于此值触发递归"""
    
    min_result_quality: float = 0.5
    """最小结果质量分数（0-1），用于判断是否需要更深层检索"""
    
    # 问题生成参数
    generate_sub_questions: bool = True
    """是否自动生成子问题进行递归检索"""
    
    num_sub_questions: int = 2
    """每层生成的子问题数量"""
    
    # 重排序参数
    enable_reranking: bool = True
    """是否在各层进行重排序"""
    
    rerank_method: str = "cosine"
    """重排序方法：cosine, bm25, cross_encoder"""
    
    # 结果合并参数
    merge_strategy: str = "weighted_dedup"
    """结果合并策略：weighted_dedup, union, intersection"""
    
    deduplication_threshold: float = 0.85
    """相似度高于此值的结果视为重复"""
    
    # 日志和调试
    enable_logging: bool = True
    """是否记录递归检索的详细过程"""
    
    debug_mode: bool = False
    """调试模式，输出更详细的日志"""


# 预设配置
class RecursiveRetrieverPresets:
    """预设配置"""
    
    @staticmethod
    def light() -> RecursiveRetrieverConfig:
        """轻量级配置 - 用于实时应用"""
        return RecursiveRetrieverConfig(
            enable_recursion=True,
            max_recursion_depth=2,
            initial_k=5,
            intermediate_k=3,
            final_k=3,
            num_sub_questions=1,
            min_confidence_score=0.5,
        )
    
    @staticmethod
    def balanced() -> RecursiveRetrieverConfig:
        """均衡配置 - 默认推荐"""
        return RecursiveRetrieverConfig(
            enable_recursion=True,
            max_recursion_depth=3,
            initial_k=10,
            intermediate_k=5,
            final_k=5,
            num_sub_questions=2,
            min_confidence_score=0.6,
        )
    
    @staticmethod
    def deep() -> RecursiveRetrieverConfig:
        """深度挖掘配置 - 用于复杂问题"""
        return RecursiveRetrieverConfig(
            enable_recursion=True,
            max_recursion_depth=4,
            initial_k=15,
            intermediate_k=8,
            final_k=5,
            num_sub_questions=3,
            min_confidence_score=0.7,
        )
    
    @staticmethod
    def single_layer() -> RecursiveRetrieverConfig:
        """单层配置 - 仅进行初始检索"""
        return RecursiveRetrieverConfig(
            enable_recursion=False,
            max_recursion_depth=1,
            initial_k=5,
        )
