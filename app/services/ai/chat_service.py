import aiohttp
import dashscope
from typing import List, Dict, Any
import logging
from fastapi import Depends

from app.services.ai.search_service import SearchService, get_search_service
from app.services.ai.embedding_service import EmbeddingService, get_embedding_service

from app.core.config import config

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self, search_service: SearchService, embedding_service: EmbeddingService
    ):
        self.search_service = search_service
        self.embedding_service = embedding_service

        self.api_key = config.DASHSCOPE_API_KEY
        self.base_url = config.DASHSCOPE_BASE_URL
        self.app_base_url = config.DASHSCOPE_APP_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.app_id = config.DASHSCOPE_APP_ID
        self.timeout = aiohttp.ClientTimeout(total=config.API_TIMEOUT)

    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            data = {
                "model": config.CHAT_MODEL,
                "messages": messages,
                "stream": False,
                "max_tokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.7),
            }

            logger.info(f"Calling DashScope API with model: {config.CHAT_MODEL}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Chat API error {response.status}: {error_text}"
                        )
        except Exception as e:
            logger.error(f"Chat API error: {e}")
            raise Exception(f"Failed to get chat completion: {str(e)}")

    async def chat_completion_with_search(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Dict[str, Any]:
        try:
            enable_search = kwargs.get("enable_search", False)
            use_deep_search = kwargs.get("use_deep_search", True)

            if enable_search:
                user_query = messages[-1]["content"]

                if use_deep_search:
                    logger.info("Calling Deep Search Agent...")
                    return await self.search_service.deep_search(user_query, **kwargs)
                else:
                    logger.info("Calling Bailian App API for web search...")
                    return await self.search_service._call_bailian_app(
                        user_query, **kwargs
                    )
            else:
                content = await self.chat_completion(messages, **kwargs)
                return {
                    "content": content,
                    "search_used": False,
                    "deep_search": False,
                    "success": True,
                }

        except Exception as e:
            logger.error(f"Web search API error: {e}")

            content = await self.chat_completion(messages, **kwargs)
            return {
                "content": f"Web search encountered an issue, switched to model knowledge:\n\n{content}",
                "search_used": False,
                "deep_search": False,
                "success": True,
                "error": str(e),
            }


# 注入工厂
def get_chat_service(
    search_service: SearchService = Depends(get_search_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> ChatService:
    # FastAPI 会先去跑 get_search_service 和 get_embedding_service
    # 拿到实例后，再喂给 ChatService
    return ChatService(search_service, embedding_service)
