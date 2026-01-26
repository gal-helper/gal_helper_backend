import json
import logging
import aiohttp
from typing import List, Dict, Any
from app.core.config import config

import dashscope

from app.core import config

logger = logging.getLogger(__name__)

class SearchService:
    def __init__(self):
        self.api_key = config.DASHSCOPE_API_KEY
        self.base_url = config.DASHSCOPE_BASE_URL
        self.app_base_url = config.DASHSCOPE_APP_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.app_id = config.DASHSCOPE_APP_ID
        self.deep_search_agent_id = config.DEEP_SEARCH_AGENT_ID
        self.deep_search_agent_version = config.DEEP_SEARCH_AGENT_VERSION

        dashscope.api_key = self.api_key

    async def _call_bailian_app(self, query: str, **kwargs) -> Dict[str, Any]:
        app_api_url = f"{self.app_base_url}/apps/{self.app_id}/completion"

        data = {
            "input": {
                "messages": [{"role": "user", "content": query}]
            },
            "parameters": {
                "max_tokens": kwargs.get("max_tokens", 2000),
                "temperature": kwargs.get("temperature", 0.7)
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    app_api_url,
                    headers=self.headers,
                    json=data,
                    timeout=config.APP_API_TIMEOUT
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("output") and result["output"].get("choices"):
                        content = result["output"]["choices"][0]["message"]["content"]
                        return {
                            "content": content,
                            "search_used": True,
                            "success": True
                        }
                    else:
                        logger.error("Bailian App response format error")
                        raise Exception("Bailian App response format error")
                else:
                    error_text = await response.text()
                    error_msg = f"Bailian App API error {response.status}"

                    try:
                        error_json = json.loads(error_text)
                        if "message" in error_json:
                            error_msg = f"{error_msg}: {error_json['message']}"
                        if "code" in error_json:
                            error_code = error_json['code']
                            if error_code == "InvalidApiKey":
                                raise Exception("Invalid API key or insufficient permissions")
                            elif error_code == "QuotaExhausted":
                                raise Exception("API quota exhausted")
                            elif error_code == "InvalidParameter":
                                raise Exception("Invalid app ID or configuration")
                    except:
                        pass

                    raise Exception(error_msg)

    async def _call_deep_search_agent(self, query: str, **kwargs) -> Dict[str, Any]:
        deep_search_url = f"{self.app_base_url}/v2/apps/deep-search-agent/chat/completions"

        data = {
            "input": {
                "messages": [{"role": "user", "content": query}]
            },
            "parameters": {
                "agent_options": {
                    "agent_id": self.deep_search_agent_id,
                    "agent_version": self.deep_search_agent_version
                }
            },
            "stream": False
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    deep_search_url,
                    headers=self.headers,
                    json=data,
                    timeout=config.APP_API_TIMEOUT
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("output") and result["output"].get("choices"):
                        content = result["output"]["choices"][0]["message"]["content"]
                        return {
                            "content": content,
                            "search_used": True,
                            "deep_search": True,
                            "success": True
                        }
                    else:
                        logger.error("Deep Search Agent response format error")
                        raise Exception("Deep Search Agent response format error")
                else:
                    error_text = await response.text()
                    error_msg = f"Deep Search Agent API error {response.status}"

                    try:
                        error_json = json.loads(error_text)
                        if "message" in error_json:
                            error_msg = f"{error_msg}: {error_json['message']}"
                        if "code" in error_json:
                            error_code = error_json['code']
                            if error_code in ["InvalidApiKey", "AccessDenied"]:
                                raise Exception("No Deep Search permission for this API key")
                            elif error_code == "QuotaExhausted":
                                raise Exception("Deep Search quota exhausted")
                            elif error_code == "InvalidParameter":
                                raise Exception("Invalid Deep Search agent configuration")
                    except:
                        pass

                    raise Exception(error_msg)

    async def deep_search(self, query: str, **kwargs) -> Dict[str, Any]:

        logger.info(f"Executing Deep Search for query: {query[:100]}...")

        try:
            return await self._call_deep_search_agent(query, **kwargs)
        except Exception as e:
            logger.error(f"Deep Search Agent failed: {e}")

            try:
                fallback_result = await self._call_bailian_app(query, **kwargs)
                logger.info("Fell back to regular Bailian App search")
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"All search methods failed: {fallback_error}")

                messages = [{"role": "user", "content": query}]
                content = await self.chat_completion(messages, **kwargs)

                return {
                    "content": f"Search services encountered issues, using model knowledge:\n\n{content}",
                    "search_used": False,
                    "deep_search": False,
                    "success": True,
                    "error": str(e)
                }


# 注入工厂
def get_search_service(

) -> SearchService:
    return SearchService()