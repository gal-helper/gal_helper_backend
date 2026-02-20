# -*- coding: utf-8 -*-
"""
文档自动标签生成工具
使用 DeepSeek API 进行 AI 智能标签化
"""

import json
import logging
import hashlib
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import aiohttp
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)


class DocumentTagger:
    """
    使用 DeepSeek API 进行文档自动标签化
    
    特性：
    - 调用 DeepSeek 生成分类/领域/难度等多维标签
    - 本地缓存避免重复调用
    - 支持批量标签化
    - 自动重试和错误处理
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        cache_ttl: int = 86400,  # 24小时缓存
        max_retries: int = 3,
    ):
        """初始化标签生成器"""
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        
        # 内存缓存：文档哈希 -> 标签
        self._cache: Dict[str, tuple] = {}  # (tags, timestamp)
    
    @staticmethod
    def _hash_content(text: str) -> str:
        """计算文本哈希值用于缓存"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _is_cached(self, content_hash: str) -> bool:
        """检查缓存是否仍有效"""
        if content_hash not in self._cache:
            return False
        
        tags, timestamp = self._cache[content_hash]
        if datetime.now() - timestamp > timedelta(seconds=self.cache_ttl):
            del self._cache[content_hash]
            return False
        
        return True
    
    async def generate_tags(
        self,
        title: str,
        content: str,
        content_type: str = "text"
    ) -> Dict[str, any]:
        """
        为文档生成多维标签
        
        返回结构：
        {
            "categories": ["技术", "教程"],  # 文档分类
            "domains": ["深度学习", "NLP"],  # 应用领域
            "difficulty": "中级",  # 难度：入门/基础/中级/高级/专家
            "importance": 0.85,  # 重要性：0-1
            "auto_tags": ["语言模型", "Transformer"],  # AI生成的标签
            "custom_tags": [],  # 用户自定义标签
            "language": "zh",  # 语言：zh/en
            "quality_score": 0.9,  # 内容质量评分
        }
        """
        
        # 检查缓存
        content_hash = self._hash_content(title + content)
        if self._is_cached(content_hash):
            logger.info(f"使用缓存标签（内容哈希：{content_hash}）")
            return self._cache[content_hash][0]
        
        # 调用 DeepSeek API
        tags = await self._call_deepseek_api(title, content, content_type)
        
        # 存储到缓存
        self._cache[content_hash] = (tags, datetime.now())
        
        return tags
    
    async def _call_deepseek_api(
        self,
        title: str,
        content: str,
        content_type: str
    ) -> Dict[str, any]:
        """
        调用 DeepSeek API 生成标签
        
        使用结构化提示确保返回有效的 JSON
        """
        
        # 截断长内容（只取前2000字用于分析）
        truncated_content = content[:2000]
        
        prompt = f"""请分析以下文档并生成结构化标签。必须返回有效的JSON格式。

文档类型: {content_type}
标题: {title}
内容摘要: {truncated_content}

请生成以下标签，返回JSON格式（必须包含所有字段）：

{{
    "categories": ["string"],  # 文档分类，3-5个，如：技术、教程、案例等
    "domains": ["string"],  # 应用领域，2-4个，如：NLP、计算机视觉等
    "difficulty": "string",  # 难度级别：入门/基础/中级/高级/专家 中选一个
    "importance": 0.85,  # 重要性分数 0-1
    "auto_tags": ["string"],  # AI自动生成的标签，3-5个关键词
    "custom_tags": [],  # 保持为空
    "language": "zh",  # 语言
    "quality_score": 0.85  # 内容质量评分 0-1
}}

注意：必须返回有效的JSON，不要包含markdown代码块。"""
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    
                    payload = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "你是一个专业的文档分类和标签化专家。"},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,  # 降低温度提高确定性
                        "max_tokens": 500,
                        "response_format": {"type": "json_object"}  # 强制 JSON 返回
                    }
                    
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            response_text = data["choices"][0]["message"]["content"]
                            
                            # 解析 JSON
                            tags = json.loads(response_text)
                            
                            # 验证必要字段
                            required_fields = [
                                "categories", "domains", "difficulty",
                                "importance", "auto_tags", "language", "quality_score"
                            ]
                            for field in required_fields:
                                if field not in tags:
                                    logger.warning(f"缺少必要字段: {field}，使用默认值")
                                    tags = self._get_default_tags(title)
                                    break
                            
                            logger.info(f"成功生成标签（第{attempt+1}次尝试）")
                            return tags
                        
                        elif resp.status == 429:
                            # 频率限制
                            wait_time = (attempt + 1) * 2
                            logger.warning(f"API速率限制，等待 {wait_time}s 后重试...")
                            await asyncio.sleep(wait_time)
                        
                        else:
                            error_text = await resp.text()
                            logger.error(f"API错误 {resp.status}: {error_text}")
                            if attempt == self.max_retries - 1:
                                return self._get_default_tags(title)
            
            except asyncio.TimeoutError:
                logger.warning(f"API超时（第{attempt+1}次尝试）")
                if attempt == self.max_retries - 1:
                    return self._get_default_tags(title)
                await asyncio.sleep((attempt + 1) * 2)
            
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {e}")
                if attempt == self.max_retries - 1:
                    return self._get_default_tags(title)
            
            except Exception as e:
                logger.error(f"生成标签失败: {e}")
                if attempt == self.max_retries - 1:
                    return self._get_default_tags(title)
        
        return self._get_default_tags(title)
    
    @staticmethod
    def _get_default_tags(title: str) -> Dict[str, any]:
        """
        生成默认标签（当 API 失败时）
        """
        return {
            "categories": ["未分类"],
            "domains": ["通用"],
            "difficulty": "中级",
            "importance": 0.5,
            "auto_tags": [],
            "custom_tags": [],
            "language": "zh",
            "quality_score": 0.5,
        }
    
    async def batch_generate_tags(
        self,
        documents: List[tuple]  # List[（title, content, content_type），...]
    ) -> List[Dict[str, any]]:
        """
        批量生成标签
        
        示例：
        docs = [
            ("标题1", "内容1", "text"),
            ("标题2", "内容2", "pdf"),
        ]
        tags = await tagger.batch_generate_tags(docs)
        """
        
        tasks = []
        for title, content, content_type in documents:
            task = self.generate_tags(title, content, content_type)
            tasks.append(task)
            
            # 避免并发数过高（最多10个并发）
            if len(tasks) >= 10:
                results = await asyncio.gather(*tasks)
                for result in results:
                    yield result
                tasks = []
        
        # 处理剩余任务
        if tasks:
            results = await asyncio.gather(*tasks)
            for result in results:
                yield result
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.info("标签缓存已清空")
    
    def cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)


# 单例实例
_tagger_instance: Optional[DocumentTagger] = None


def get_tagger(api_key: str, base_url: str = "https://api.deepseek.com/v1") -> DocumentTagger:
    """获取或创建全局标签生成器实例"""
    global _tagger_instance
    
    if _tagger_instance is None:
        _tagger_instance = DocumentTagger(api_key=api_key, base_url=base_url)
    
    return _tagger_instance


async def tag_document(
    title: str,
    content: str,
    content_type: str = "text",
    api_key: Optional[str] = None
) -> Dict[str, any]:
    """
    便捷函数：直接为文档生成标签
    
    使用：
    tags = await tag_document("标题", "内容")
    """
    
    if api_key is None:
        # 从环境变量或配置获取
        import os
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.error("未设置 DEEPSEEK_API_KEY")
            return DocumentTagger._get_default_tags(title)
    
    tagger = get_tagger(api_key)
    return await tagger.generate_tags(title, content, content_type)
