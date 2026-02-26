# -*- coding: utf-8 -*-
"""
统一的文档存储模型
支持：向量 + 关键词 + 标签 三维混合检索
内存索引 + 磁盘存储分离设计
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import DateTime, func, Index, Text, Float, String
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import Integer, Boolean

try:
    # 尝试从 pgvector.sqlalchemy 导入 Vector（推荐）
    from pgvector.sqlalchemy import Vector
except ImportError:
    # 降级方案：如果 pgvector 包未安装，使用通用的 ARRAY
    from sqlalchemy import ARRAY as Vector

class Base(DeclarativeBase):
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        comment="创建时间"
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
        comment="更新时间"
    )


class Document(Base):
    """
    统一文档表 - 支持向量+关键词+标签混合检索
    
    设计原则：
    - 向量索引（memory）+ 向量数据（disk）分离
    - 标签索引（memory）+ 标签数据（disk）分离
    - 关键词数据放PostgreSQL
    """
    __tablename__ = "ai_documents"
    
    # ===== 基础字段 =====
    id: Mapped[int] = mapped_column(Integer, primary_key=True, comment="文档ID")
    doc_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, comment="文档哈希值，用于去重")
    
    # ===== 内容字段（磁盘存储） =====
    title: Mapped[str] = mapped_column(String(512), nullable=False, comment="文档标题")
    content: Mapped[str] = mapped_column(Text, nullable=False, comment="完整文档内容")
    content_type: Mapped[str] = mapped_column(String(50), comment="内容类型：PDF/TXT/MD等")
    source_url: Mapped[Optional[str]] = mapped_column(String(1024), comment="来源URL")
    
    # ===== 向量字段（内存索引 + 磁盘存储） =====
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(1536), comment="文档向量（1536维）")
    embedding_model: Mapped[str] = mapped_column(String(100), default="nomic-embed-text", comment="向量模型名称")
    
    # ===== 关键词字段（PostgreSQL 存储） =====
    # 由 split_keywords() 方法自动生成
    keywords: Mapped[Optional[List[str]]] = mapped_column(
        ARRAY(String), 
        comment="关键词列表（用于关键词检索）"
    )
    
    # ===== 标签字段（内存索引 + 磁盘存储） =====
    # 结构示例：
    # {
    #   "categories": ["技术", "教程"],
    #   "domains": ["深度学习", "NLP"],
    #   "difficulty": "中级",
    #   "importance": 0.85,
    #   "auto_tags": ["语言模型", "transformer"],
    #   "custom_tags": ["推荐", "高质量"]
    # }
    tags: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        default={},
        comment="结构化标签：分类/领域/难度/重要性/自动标签/自定义标签"
    )
    
    # ===== 元数据字段 =====
    extra_metadata: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        default={},
        comment="额外元数据：作者/来源/版本等"
    )
    
    # ===== 检索辅助字段 =====
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否已建立向量索引")
    is_tagged: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否已自动标签化")
    retrieval_count: Mapped[int] = mapped_column(Integer, default=0, comment="被检索次数")
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0, comment="平均相关性得分")
    
    # ===== 索引定义 =====
    __table_args__ = (
        # 向量相似度搜索索引
        Index('ix_document_embedding', 'embedding', postgresql_using='ivfflat'),
        
        # 标签快速查询索引
        Index('ix_document_tags', 'tags', postgresql_using='gin'),
        
        # 关键词全文搜索索引
        Index('ix_document_keywords', 'keywords', postgresql_using='gin'),
        
        # 复合索引：用于常见查询组合
        Index('ix_document_is_indexed_is_tagged', 'is_indexed', 'is_tagged'),
    )
    
    def __repr__(self):
        return f"<Document(id={self.id}, title={self.title[:30]}..., is_indexed={self.is_indexed}, is_tagged={self.is_tagged})>"
    
    def split_keywords(self, max_keywords: int = 10) -> List[str]:
        """
        从标题和内容中提取关键词
        
        优先级：
        1. 标签中的 auto_tags
        2. 内容中的高频词
        3. 标题中的词
        """
        if not self.keywords:
            self.keywords = []
        
        # 从标签中提取自动标签
        if self.tags and isinstance(self.tags, dict):
            auto_tags = self.tags.get("auto_tags", [])
            self.keywords = list(auto_tags[:max_keywords])
        
        return self.keywords


class DocumentTagCache(Base):
    """
    文档标签缓存表 - 用于高速标签检索
    
    内存索引放这里，减少对 documents 表的 JSONB 查询
    """
    __tablename__ = "ai_document_tag_cache"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, nullable=False, comment="关联的文档ID")
    
    # 按类型分类缓存
    categories: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), comment="分类标签")
    domains: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), comment="领域标签")
    auto_tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), comment="自动标签")
    custom_tags: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String), comment="自定义标签")
    
    difficulty: Mapped[Optional[str]] = mapped_column(String(50), comment="难度级别")
    importance: Mapped[float] = mapped_column(Float, default=0.5, comment="重要性分数")
    
    # 缓存命中率跟踪
    hit_count: Mapped[int] = mapped_column(Integer, default=0, comment="缓存命中次数")
    
    __table_args__ = (
        # 快速查询索引
        Index('ix_tag_cache_document_id', 'document_id'),
        Index('ix_tag_cache_categories', 'categories', postgresql_using='gin'),
        Index('ix_tag_cache_domains', 'domains', postgresql_using='gin'),
        Index('ix_tag_cache_auto_tags', 'auto_tags', postgresql_using='gin'),
    )
    
    def __repr__(self):
        return f"<DocumentTagCache(document_id={self.document_id}, categories={self.categories})>"


class DocumentEmbeddingIndex(Base):
    """
    向量索引缓存表 - 用于高速向量检索
    
    保存向量ID映射，支持快速向量相似度搜索
    """
    __tablename__ = "ai_document_embedding_index"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(Integer, nullable=False, unique=True, comment="关联的文档ID")
    
    # 向量ID (PgVector 自动生成)
    vector_id: Mapped[Optional[str]] = mapped_column(String(256), comment="向量ID，用于检索")
    
    # 向量统计信息
    embedding_dim: Mapped[int] = mapped_column(Integer, default=1536, comment="向量维度")
    norm: Mapped[float] = mapped_column(Float, comment="向量的L2范数")
    
    # 性能指标
    search_count: Mapped[int] = mapped_column(Integer, default=0, comment="被搜索次数")
    avg_similarity: Mapped[float] = mapped_column(Float, default=0.0, comment="平均相似度")
    
    __table_args__ = (
        Index('ix_embedding_index_document_id', 'document_id'),
        Index('ix_embedding_index_vector_id', 'vector_id'),
    )
    
    def __repr__(self):
        return f"<DocumentEmbeddingIndex(document_id={self.document_id}, vector_id={self.vector_id})>"
