"""
LangChain 核心模块管理
单表模式：所有文档存储到 ai_documents 表
"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# LangChain 0.1.x - 使用 langchain_openai
try:
    from langchain_openai import ChatOpenAI
    _has_chat_openai = True
except ImportError:
    _has_chat_openai = False
    ChatOpenAI = None

try:
    from langchain_community.embeddings import OllamaEmbeddings
    _has_ollama_embeddings = True
except ImportError:
    _has_ollama_embeddings = False
    OllamaEmbeddings = None

from app.core.config import config
from app.core.db import db_initializer


class SimpleLangchainManager:
    """Langchain 管理器 - 单表模式"""
    
    def __init__(self):
        self._chatModel = None
        self._embeddingModel = None
        self._vectorstore = None
        self._initialized = False

    async def initialize(self):
        """初始化 Langchain 组件"""
        if self._initialized:
            return

        logger.info("Initializing Langchain components...")
        
        # 确保数据库已初始化
        await db_initializer.initialize()
        
        # 初始化模型
        self._init_models()
        
        self._initialized = True
        logger.info("✅ Langchain components initialized successfully")

    def _init_models(self):
        """初始化模型"""
        # Chat 模型 - DeepSeek (使用 OpenAI 兼容接口)
        if ChatOpenAI and config.CHAT_MODEL_NAME:
            try:
                logger.info(f"Connecting to chat model: {config.CHAT_MODEL_NAME}")
                self._chatModel = ChatOpenAI(
                    model=config.CHAT_MODEL_NAME,
                    base_url=config.CHAT_MODEL_BASE_URL,
                    api_key=config.CHAT_MODEL_API_KEY or "sk-xxx",
                    temperature=0.7,
                    max_tokens=2000,
                    request_timeout=60,
                    max_retries=3,
                )
                logger.info(f"✅ Chat model connected: {config.CHAT_MODEL_NAME}")
            except Exception as e:
                logger.warning(f"Failed to connect chat model: {e}")
                self._chatModel = None

        # Embedding 模型 - Ollama
        if OllamaEmbeddings and config.BASE_EMBEDDING_MODEL_NAME:
            try:
                logger.info(f"Connecting to embedding model: {config.BASE_EMBEDDING_MODEL_NAME}")
                self._embeddingModel = OllamaEmbeddings(
                    model=config.BASE_EMBEDDING_MODEL_NAME,
                    base_url=config.BASE_EMBEDDING_MODEL_BASE_URL,
                )
                logger.info(f"✅ Embedding model connected: {config.BASE_EMBEDDING_MODEL_NAME}")
            except Exception as e:
                logger.warning(f"Failed to connect embedding model: {e}")
                self._embeddingModel = None

    def get_chat_model(self):
        """获取 Chat 模型"""
        if not self._chatModel:
            logger.warning("Chat model not available")
            return None
        return self._chatModel

    def get_base_embeddings(self):
        """获取 Embedding 模型"""
        if not self._embeddingModel:
            logger.warning("Embedding model not available")
            return None
        return self._embeddingModel

    def get_vectorstore(self):
        """获取向量存储 - 单表模式"""
        if self._vectorstore:
            return self._vectorstore
            
        try:
            from langchain_postgres import PGVector
            
            connection_string = config.LANGCHAIN_DATABASE_URL
            if not connection_string:
                logger.error("❌ LANGCHAIN_DATABASE_URL not configured in .env")
                return None
            
            embeddings = self.get_base_embeddings()
            if not embeddings:
                logger.error("❌ Embedding model not available")
                return None
            
            # 单表：collection_name 固定为 ai_documents
            self._vectorstore = PGVector(
                embeddings=embeddings,
                connection=connection_string,
                collection_name="ai_documents",
            )
            
            logger.info("✅ Connected to PostgreSQL vectorstore (single table)")
            return self._vectorstore
        
        except ImportError as e:
            logger.error(f"❌ langchain_postgres not installed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to initialize vectorstore: {type(e).__name__}: {e}")
            return None

    def get_checkpointer(self):
        """获取检查点器"""
        logger.warning("Checkpointer not available")
        return None


# 全局管理器实例
langchain_manager = SimpleLangchainManager()
