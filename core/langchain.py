# app/core/langchain.py
"""
LangChain 核心模块管理
支持 Chat、Embeddings 等功能
简化版本，避免复杂的依赖链
"""
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# 导入 LangChain 核心模块 - 使用 try/except 处理缺失模块
try:
    from langchain.chat_models import init_chat_model
except ImportError:
    init_chat_model = None

try:
    from langchain.embeddings import init_embeddings
except ImportError:
    init_embeddings = None

from app.core.config import config
from app.core.db import db_initializer


class SimpleLangchainManager:
    """
    简化的 Langchain 管理器
    避免复杂的框架依赖链
    """
    def __init__(self):
        self._chatModel = None
        self._embeddingModel = None
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
        # Chat 模型 - DeepSeek
        if init_chat_model:
            try:
                logger.info(f"Connecting to chat model (DeepSeek): {config.CHAT_MODEL_NAME}")
                # DeepSeek 使用 OpenAI 兼容的 API
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
                logger.info(f"✅ Chat model (DeepSeek) connected: {config.CHAT_MODEL_NAME}")
            except Exception as e:
                logger.warning(f"Failed to connect chat model (DeepSeek): {e}")
                self._chatModel = None
        else:
            logger.warning("LangChain chat_models not available")

        # Embedding 模型 - Ollama
        if init_embeddings:
            try:
                logger.info(f"Connecting to embedding model (Ollama): {config.BASE_EMBEDDING_MODEL_NAME}")
                # Ollama 使用 OpenAI 兼容格式
                self._embeddingModel = init_embeddings(
                    model=f"openai:{config.BASE_EMBEDDING_MODEL_NAME}",
                    base_url=config.BASE_EMBEDDING_MODEL_BASE_URL,
                    api_key=config.BASE_EMBEDDING_API_KEY or "ollama",
                    timeout=60,
                    max_retries=3,
                )
                logger.info(f"✅ Embedding model (Ollama) connected: {config.BASE_EMBEDDING_MODEL_NAME}")
            except Exception as e:
                logger.warning(f"Failed to connect embedding model (Ollama): {e}")
                self._embeddingModel = None
        else:
            logger.warning("LangChain embeddings not available")

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
        """获取向量存储 - 直接连接数据库的 ai_documents 表"""
        try:
            from langchain_postgres import PGVector
            
            # 使用配置中的数据库连接字符串
            connection_string = config.LANGCHAIN_DATABASE_URL
            if not connection_string:
                logger.error("❌ LANGCHAIN_DATABASE_URL not configured in .env")
                return None
            
            # 获取嵌入模型
            embedding_function = self.get_base_embeddings()
            if not embedding_function:
                logger.error("❌ Embedding model not available - cannot initialize vectorstore")
                return None
            
            # 创建 PGVector 实例，连接到 ai_documents 表
            vectorstore = PGVector(
                connection_string=connection_string,
                embedding_function=embedding_function,
                collection_name="ai_documents",
                use_jsonb=True,
            )
            
            logger.info("✅ Connected to PostgreSQL vectorstore (ai_documents table)")
            return vectorstore
        
        except ImportError as e:
            logger.error(f"❌ langchain_postgres not installed: {e}")
            logger.info("   Install with: pip install langchain-postgres")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to initialize vectorstore: {type(e).__name__}: {e}")
            logger.debug(f"   Connection string: {config.LANGCHAIN_DATABASE_URL[:50]}...")
            return None

    async def async_get_vectorstore_for_table(self, table_name: str):
        """获取特定表的向量存储（多表支持）"""
        try:
            from langchain_postgres import PGVector
            
            # 如果表名为空，使用默认的 ai_documents
            if not table_name:
                return self.get_vectorstore()
            
            connection_string = config.LANGCHAIN_DATABASE_URL
            if not connection_string:
                logger.error(f"❌ LANGCHAIN_DATABASE_URL not configured")
                return None
            
            embedding_function = self.get_base_embeddings()
            if not embedding_function:
                logger.error(f"❌ Embedding model not available for table: {table_name}")
                return None
            
            # 为不同的表创建不同的 collection
            vectorstore = PGVector(
                connection_string=connection_string,
                embedding_function=embedding_function,
                collection_name=table_name,
                use_jsonb=True,
            )
            
            logger.info(f"✅ Connected to vectorstore for table: {table_name}")
            return vectorstore
        
        except ImportError as e:
            logger.error(f"❌ langchain_postgres not installed: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to initialize vectorstore for table '{table_name}': {e}")
            return None

    def get_checkpointer(self):
        """获取检查点器（返回 None，不支持对话历史保存）"""
        logger.warning("Checkpointer not available in simplified manager")
        return None


# 创建全局管理器实例
langchain_manager = SimpleLangchainManager()