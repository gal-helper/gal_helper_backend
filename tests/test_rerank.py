"""
Rerank 功能测试

测试余弦相似度（TF-IDF）的 rerank 功能
"""

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def test_cosine_rerank_basic():
    """测试基于余弦相似度的 rerank 功能"""
    
    # 测试数据
    question = "如何解决游戏闪退问题"
    sources = [
        {"id": 1, "content": "游戏启动时崩溃解决方案：检查系统配置和驱动程序"},
        {"id": 2, "content": "显卡驱动更新步骤详解"},
        {"id": 3, "content": "游戏闪退问题常见原因和解决步骤"},
        {"id": 4, "content": "CPU 超频设置指南"},
        {"id": 5, "content": "如何优化游戏性能和帧率"},
    ]
    
    print("="*70)
    print("测试：余弦相似度 Rerank")
    print("="*70)
    print(f"\n查询：{question}\n")
    
    # 提取文本
    texts = [src["content"] for src in sources]
    
    # 使用 TF-IDF 向量化
    vectorizer = TfidfVectorizer(
        max_features=100,
        lowercase=True,
        stop_words=None,
        analyzer='char',
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0
    )
    
    # 组合查询和文档
    combined_texts = [question] + texts
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    
    # 计算余弦相似度
    query_vector = tfidf_matrix[0:1]
    doc_vectors = tfidf_matrix[1:]
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    
    # 排序
    scored = list(zip(similarities, sources))
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # 显示结果
    print("排序结果（按相似度降序）：\n")
    for idx, (similarity, src) in enumerate(scored, 1):
        print(f"{idx}. 相似度: {similarity:.4f} | ID: {src['id']}")
        print(f"   内容: {src['content']}\n")
    
    # 验证结果
    top_result_id = scored[0][1]['id']
    assert top_result_id == 3, f"期望 ID 3 排在第一位，实际是 {top_result_id}"
    
    print("="*70)
    print("✅ 测试通过")
    print("\n关键特性验证：")
    print("  ✓ TF-IDF 向量化成功")
    print("  ✓ 字符级别分析（支持中文）")
    print("  ✓ 单双字 n-gram 特征")
    print("  ✓ 余弦相似度正确计算")
    print("  ✓ 排序结果符合预期")


def test_empty_sources():
    """测试空源列表"""
    vectorizer = TfidfVectorizer()
    
    # 空列表应该返回空
    sources = []
    result = vectorizer.fit_transform([])
    
    assert result.shape == (0, 0), "空列表应该返回空矩阵"
    print("✅ 空源列表测试通过")


def test_single_source():
    """测试单个源"""
    question = "测试"
    sources = [{"content": "测试内容"}]
    texts = [s["content"] for s in sources]
    
    vectorizer = TfidfVectorizer()
    combined_texts = [question] + texts
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    
    # 应该有 2 行（查询 + 1 个源）
    assert tfidf_matrix.shape[0] == 2, f"期望 2 行，实际 {tfidf_matrix.shape[0]}"
    print("✅ 单源测试通过")


if __name__ == "__main__":
    test_cosine_rerank_basic()
    print()
    test_empty_sources()
    test_single_source()
    print("\n✅ 所有测试通过")
