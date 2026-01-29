from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.core.config import config
from app.core.db import async_db_manager
import logging

logger = logging.getLogger(__name__)

class LangchainManager:
    def __init__(self):
        self._chatModel: Optional[BaseChatModel] = None
        self._checkpointer: Optional[AsyncPostgresSaver] = None

    async def init_langchain_manager(self):
        """ 初始化langchain管理器 """
        await self._init_base_chat_model()
        self._checkpointer = await self.get_checkpointer()

    async def _init_base_chat_model(self):
        """ 初始化基本的chatModel """
        if self._chatModel:
            return
        try:
            self._chatModel = init_chat_model(
                model=config.CHAT_MODEL_NAME,
                model_provider="openai",
                base_url=config.CHAT_MODEL_BASE_URL,
                api_key=config.CHAT_MODEL_API_KEY,
                temperature=0.7,
                max_tokens=2000,
                timeout=None,
                max_retries=2
            )
            logger.info(f"Connected to chat model: {config.CHAT_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to chat model: {e}")
            raise e

    async def get_base_chat_model(self):
        """ 获取基本的chatModel """
        if not self._chatModel:
            await self._init_base_chat_model()
        return self._chatModel

    async def get_checkpointer(self):
        """ 获取postgresSQL管理的checkpointer """
        if not self._checkpointer:
            # 这里的获取方式非常干净
            pool = async_db_manager.get_raw_pool()
            self._checkpointer = AsyncPostgresSaver(pool)
            # setup 会在数据库创建 langgraph 需要的表 只运行一次！！！
            await self._checkpointer.setup()
        return self._checkpointer


langchain_manager = LangchainManager()
