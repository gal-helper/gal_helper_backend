# app/core/langchain.py
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from langchain_postgres.v2.engine import PGEngine

from app.core.config import config
from app.core.db import async_db_manager, langchain_pool, db_initializer
import logging

logger = logging.getLogger(__name__)


class LangchainManager:
    def __init__(self):
        self._chatModel: Optional[BaseChatModel] = None
        self._embeddingModel: Optional[Embeddings] = None
        self._checkpointer: Optional[AsyncPostgresSaver] = None
        self._vectorstore: Optional[AsyncPGVectorStore] = None
        self._initialized = False

    async def initialize(self):
        """初始化所有 Langchain 组件"""
        if self._initialized:
            return

        logger.info("Initializing Langchain components...")

        # 1. 确保数据库已初始化
        await db_initializer.initialize()

        # 2. 初始化各组件
        self._init_models()
        await self._init_checkpointer()
        await self._init_vectorstore()

        self._initialized = True
        logger.info("Langchain components initialized successfully")

    def _init_models(self):
        """初始化模型"""
        # Chat 模型
        try:
            logger.info(f"Connecting to chat model: {config.CHAT_MODEL_NAME}")
            self._chatModel = init_chat_model(
                model=config.CHAT_MODEL_NAME,
                model_provider="openai",
                base_url=config.CHAT_MODEL_BASE_URL,
                api_key=config.CHAT_MODEL_API_KEY or "sk-xxx",
                temperature=0.7,
                max_tokens=2000,
                timeout=60,
                max_retries=3,
            )
            logger.info(f"✅ Chat model connected: {config.CHAT_MODEL_NAME}")
        except Exception as e:
            logger.error(f"❌ Failed to connect chat model: {e}")
            raise

        # Embedding 模型
        try:
            logger.info(f"Connecting to embedding model: {config.BASE_EMBEDDING_MODEL_NAME}")
            self._embeddingModel = init_embeddings(
                model=f"openai:{config.BASE_EMBEDDING_MODEL_NAME}",
                base_url=config.BASE_EMBEDDING_MODEL_BASE_URL,
                api_key=config.BASE_EMBEDDING_API_KEY or "sk-xxx",
                timeout=60,
                max_retries=3,
            )
            logger.info(f"✅ Embedding model connected: {config.BASE_EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"❌ Failed to connect embedding model: {e}")
            raise

    async def _init_checkpointer(self):
        """初始化检查点 - 让 LangGraph 自己管理表"""
        try:
            pool = langchain_pool.get_pool()
            self._checkpointer = AsyncPostgresSaver(conn=pool)

            # 让 LangGraph 自己创建表结构
            # 这会创建它期望的正确表结构
            await self._checkpointer.setup()

            logger.info("✅ Checkpointer initialized with auto-created tables")

        except Exception as e:
            logger.error(f"❌ Failed to initialize checkpointer: {e}")
            self._checkpointer = None

    async def _init_vectorstore(self):
        """初始化向量存储"""
        try:
            embeddings = self.get_base_embeddings()
            async_engine = async_db_manager.async_engine

            if not async_engine:
                raise RuntimeError("Async engine not initialized")

            pg_engine = PGEngine.from_engine(async_engine)

            # 明确指定所有列名，与你创建的表结构匹配
            self._vectorstore = await AsyncPGVectorStore.create(
                engine=pg_engine,
                embedding_service=embeddings,
                table_name="document_embeddings",
                schema_name="public",
                id_column="langchain_id",  # 你的 ID 列名
                content_column="document_content",  # 你的内容列名
                embedding_column="embedding",  # 你的向量列名
                metadata_json_column="langchain_metadata",  # 你的元数据列名
            )

            logger.info("✅ Vector store initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            raise

    # Getter 方法 - 所有 getter 都是同步的
    def get_chat_model(self) -> BaseChatModel:
        if not self._chatModel:
            raise RuntimeError("Chat model not initialized")
        return self._chatModel

    def get_base_embeddings(self) -> Embeddings:
        if not self._embeddingModel:
            raise RuntimeError("Embedding model not initialized")
        return self._embeddingModel

    def get_checkpointer(self) -> AsyncPostgresSaver:
        if not self._checkpointer:
            raise RuntimeError("Checkpointer not initialized")
        return self._checkpointer

    def get_vectorstore(self) -> AsyncPGVectorStore:
        if not self._vectorstore:
            raise RuntimeError("Vector store not initialized")
        return self._vectorstore


# 创建全局管理器
langchain_manager = LangchainManager()