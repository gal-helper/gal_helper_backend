from typing import Optional

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from langchain_postgres.v2.engine import PGEngine

from app.core.config import config
from app.core.db import async_db_manager, langchain_pool
import logging

logger = logging.getLogger(__name__)


class LangchainManager:
    def __init__(self):
        self._chatModel: Optional[BaseChatModel] = None
        self._embeddingModel: Optional[Embeddings] = None
        self._checkpointer: Optional[AsyncPostgresSaver] = None
        self._vectorstore: Optional[AsyncPGVectorStore] = None

    async def init_langchain_manager(self):
        """初始化langchain管理器"""
        await self._init_base_chat_model()
        self._checkpointer = await self.get_checkpointer()
        await self._init_base_embeddings()
        self._vectorstore = await self.get_vectorstore()

    async def _init_base_chat_model(self):
        """初始化基本的chatModel"""
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
                max_retries=2,
            )
            logger.info(f"Connected to chat model: {config.CHAT_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to chat model: {e}")
            raise e

    async def get_base_chat_model(self):
        """获取基本的chatModel"""
        if not self._chatModel:
            await self._init_base_chat_model()
        return self._chatModel

    async def _init_base_embeddings(self):
        """获取基本的embeddingModel"""
        if self._embeddingModel:
            return
        try:
            self._embeddingModel = init_embeddings(
                model=f"openai:{config.BASE_EMBEDDING_MODEL_NAME}",
                base_url=config.BASE_EMBEDDING_MODEL_BASE_URL,
                api_key=config.BASE_EMBEDDING_API_KEY,
                timeout=None,
                max_retries=2,
            )
            logger.info(
                f"Connected to embedding model: {config.BASE_EMBEDDING_MODEL_NAME}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to embedding model: {e}")
            raise e

    async def get_base_embeddings(self) -> Embeddings:
        """获取基本的embeddingModel"""
        if not self._embeddingModel:
            await self._init_base_embeddings()
        if self._embeddingModel is None:
            raise RuntimeError("Embedding model not initialized")
        return self._embeddingModel

    async def get_checkpointer(self) -> AsyncPostgresSaver:
        """获取postgresSQL管理的checkpointer"""
        if not self._checkpointer:
            pool = langchain_pool.get_pool()
            self._checkpointer = AsyncPostgresSaver(conn=pool)
            await self._checkpointer.setup()
        return self._checkpointer

    async def get_vectorstore(self) -> AsyncPGVectorStore:
        """获取向量存储，复用业务模块的 AsyncEngine"""
        if not self._vectorstore:
            embeddings = await self.get_base_embeddings()

            await async_db_manager.init_async_database()
            async_engine = async_db_manager.async_engine
            if not async_engine:
                raise RuntimeError("Async engine not initialized")

            pg_engine = PGEngine.from_engine(async_engine)
            self._vectorstore = await AsyncPGVectorStore.create(
                engine=pg_engine,
                embedding_service=embeddings,
                table_name="document_embeddings",
                id_column="langchain_id",
                content_column="document_content",
                embedding_column="embedding",
                metadata_json_column="langchain_metadata",
            )
            logger.info("AsyncPGVectorStore initialized (reusing ORM engine)")
        return self._vectorstore


langchain_manager = LangchainManager()
