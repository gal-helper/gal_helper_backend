import logging
import aiohttp
from typing import List
from app.core.config import config

import dashscope


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.api_key = config.DASHSCOPE_API_KEY
        self.base_url = config.DASHSCOPE_BASE_URL
        self.app_base_url = config.DASHSCOPE_APP_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.app_id = config.DASHSCOPE_APP_ID
        self.deep_search_agent_id = config.DEEP_SEARCH_AGENT_ID
        self.deep_search_agent_version = config.DEEP_SEARCH_AGENT_VERSION
        self.timeout = aiohttp.ClientTimeout(total=config.EMBEDDING_TIMEOUT)

        dashscope.api_key = self.api_key

    async def get_embedding(self, text: str) -> List[float]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json={
                        "model": config.EMBEDDING_MODEL,
                        "input": text,
                        "encoding_format": "float",
                    },
                    timeout=self.timeout,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result["data"][0]["embedding"]
                        logger.info(f"Generated embedding: {len(embedding)} dimensions")
                        return embedding
                    else:
                        error_text = await response.text()
                        raise Exception(f"API error {response.status}: {error_text}")
        except Exception as e:
            logger.error(f"Embedding API error: {e}")
            raise Exception(f"Failed to generate embedding: {str(e)}")


# 注入工厂
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
