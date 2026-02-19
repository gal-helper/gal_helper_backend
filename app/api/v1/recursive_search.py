"""
递归检索 API 端点
集成到 FastAPI 路由中，支持 REST 调用
"""

from fastapi import APIRouter, Form, HTTPException, Depends, Query
from typing import Optional
import logging
from app.services.retriever import RecursiveRetriever
from app.services.retriever.config import RecursiveRetrieverPresets
from app.core.langchain import langchain_manager

router = APIRouter(tags=["recursive-retrieval"])
logger = logging.getLogger(__name__)


@router.post("/recursive-search")
async def recursive_search(
    query: str = Form(...),
    topic: Optional[str] = Form(None),
    preset: str = Form("balanced"),
    enable_logging: bool = Form(True),
):
    """
    执行递归检索
    
    请求参数：
    - query: 搜索查询（必需）
    - topic: 主题/表名（可选，如 vectorstore_resource）
    - preset: 预设配置（light/balanced/deep/single_layer，默认 balanced）
    - enable_logging: 是否启用日志（默认 true）
    
    返回示例：
    ```json
    {
        "success": true,
        "results": [
            {
                "content": "文档内容...",
                "metadata": {...},
                "relevance_score": 0.92,
                "retrieval_depth": 2,
                "retrieval_path": ["原始查询", "子问题1"]
            }
        ],
        "report": {
            "total_results": 28,
            "final_results": 5,
            "recursion_depth_used": 3,
            "execution_time": 2.45,
            "merge_info": {"strategy": "weighted_dedup"}
        }
    }
    ```
    """
    try:
        # 选择预设
        presets = {
            "light": RecursiveRetrieverPresets.light,
            "balanced": RecursiveRetrieverPresets.balanced,
            "deep": RecursiveRetrieverPresets.deep,
            "single_layer": RecursiveRetrieverPresets.single_layer,
        }
        
        if preset not in presets:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {preset}")
        
        config = presets[preset]()
        config.enable_logging = enable_logging
        
        # 创建检索器
        vectorstore = langchain_manager.get_vectorstore()
        retriever = RecursiveRetriever(config=config, vectorstore=vectorstore)
        
        # 执行检索
        results, report = await retriever.retrieve(
            query=query,
            topic=topic,
            return_report=True,
        )
        
        return {
            "success": True,
            "results": results,
            "report": {
                "total_results": report.total_results,
                "final_results": report.final_results,
                "recursion_depth_used": report.recursion_depth_used,
                "execution_time": report.execution_time,
                "merge_info": report.merge_info,
                "retrieval_tree": report.retrieval_tree,
            }
        }
    
    except Exception as e:
        logger.error(f"递归检索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recursive-search/presets")
async def list_presets():
    """
    获取可用的预设配置列表
    
    返回示例：
    ```json
    {
        "presets": [
            {
                "name": "light",
                "description": "快速、浅层检索",
                "max_depth": 2,
                "initial_k": 5,
                "execution_time": "~1s"
            },
            ...
        ]
    }
    ```
    """
    return {
        "presets": [
            {
                "name": "light",
                "description": "Fast, shallow retrieval",
                "max_depth": 2,
                "initial_k": 5,
                "intermediate_k": 3,
                "final_k": 3,
                "num_sub_questions": 1,
                "estimated_time": "~1s",
                "accuracy": "⭐⭐",
                "best_for": "Real-time queries, simple questions"
            },
            {
                "name": "balanced",
                "description": "Recommended balanced approach",
                "max_depth": 3,
                "initial_k": 10,
                "intermediate_k": 5,
                "final_k": 5,
                "num_sub_questions": 2,
                "estimated_time": "~2-3s",
                "accuracy": "⭐⭐⭐⭐",
                "best_for": "General questions, daily use"
            },
            {
                "name": "deep",
                "description": "Deep exploration retrieval",
                "max_depth": 4,
                "initial_k": 15,
                "intermediate_k": 8,
                "final_k": 5,
                "num_sub_questions": 3,
                "estimated_time": "~4-6s",
                "accuracy": "⭐⭐⭐⭐⭐",
                "best_for": "Complex questions, research"
            },
            {
                "name": "single_layer",
                "description": "Single layer retrieval only",
                "max_depth": 1,
                "initial_k": 5,
                "estimated_time": "~0.8s",
                "accuracy": "⭐⭐",
                "best_for": "Testing, disabling recursion"
            }
        ]
    }


@router.post("/recursive-search/custom")
async def recursive_search_custom(
    query: str = Form(...),
    topic: Optional[str] = Form(None),
    max_depth: int = Form(3),
    initial_k: int = Form(10),
    intermediate_k: int = Form(5),
    final_k: int = Form(5),
    min_confidence_score: float = Form(0.6),
    num_sub_questions: int = Form(2),
    rerank_method: str = Form("cosine"),
    deduplication_threshold: float = Form(0.85),
):
    """
    使用自定义配置执行递归检索
    
    所有参数都可选，使用默认值。示例请求：
    ```
    POST /recursive-search/custom
    
    query=如何修复游戏崩溃?
    max_depth=2
    initial_k=8
    rerank_method=cross_encoder
    ```
    
    返回格式同 /recursive-search 端点
    """
    try:
        from app.services.retriever import RecursiveRetrieverConfig
        
        # 创建自定义配置
        config = RecursiveRetrieverConfig(
            enable_recursion=max_depth > 1,
            max_recursion_depth=max_depth,
            initial_k=initial_k,
            intermediate_k=intermediate_k,
            final_k=final_k,
            min_confidence_score=min_confidence_score,
            num_sub_questions=num_sub_questions,
            rerank_method=rerank_method,
            deduplication_threshold=deduplication_threshold,
        )
        
        # 创建检索器
        vectorstore = langchain_manager.get_vectorstore()
        retriever = RecursiveRetriever(config=config, vectorstore=vectorstore)
        
        # 执行检索
        results, report = await retriever.retrieve(
            query=query,
            topic=topic,
            return_report=True,
        )
        
        return {
            "success": True,
            "config_used": {
                "max_depth": max_depth,
                "initial_k": initial_k,
                "rerank_method": rerank_method,
                "deduplication_threshold": deduplication_threshold,
            },
            "results": results,
            "report": {
                "total_results": report.total_results,
                "final_results": report.final_results,
                "recursion_depth_used": report.recursion_depth_used,
                "execution_time": report.execution_time,
                "merge_info": report.merge_info,
                "retrieval_tree": report.retrieval_tree,
            }
        }
    
    except Exception as e:
        logger.error(f"自定义递归检索失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recursive-search/stats")
async def get_stats():
    """
    获取递归检索的统计信息
    
    返回示例：
    ```json
    {
        "total_searches": 1234,
        "avg_execution_time": 2.5,
        "most_used_preset": "balanced",
        "avg_recursion_depth": 2.3,
        "avg_results_collected": 18,
        "success_rate": 98.5
    }
    ```
    """
    # 这是一个占位符，实际需要添加统计日志
    return {
        "total_searches": 0,
        "avg_execution_time": 0,
        "most_used_preset": "balanced",
        "avg_recursion_depth": 0,
        "avg_results_collected": 0,
        "success_rate": 0,
        "note": "统计功能需要额外配置日志中间件"
    }
