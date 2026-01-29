from typing import Any, Dict, Literal
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Config(BaseSettings):

    # 通用的模型配置，使用openai的接口规范(基本都兼容)
    CHAT_MODEL_BASE_URL: str = ""
    CHAT_MODEL_NAME: str = ""
    CHAT_MODEL_API_KEY: str = ""

    # 通用的embedding模型配置
    BASE_EMBEDDING_MODEL_BASE_URL: str = ""
    BASE_EMBEDDING_MODEL_NAME: str = ""
    BASE_EMBEDDING_API_KEY: str = ""

    DASHSCOPE_API_KEY: str = ""
    DB_PASSWORD: str = ""
    DASHSCOPE_APP_ID: str = ""

    # 阿里的模型配置
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DASHSCOPE_APP_BASE_URL:str = "https://dashscope.aliyuncs.com/api/v1"
    EMBEDDING_MODEL: str = "text-embedding-v2"
    CHAT_MODEL: str = "qwen3-max"
    API_TIMEOUT: int = 60
    EMBEDDING_TIMEOUT: int = 30
    APP_API_TIMEOUT: int = 90

    # 数据库配置
    DB_HOST: str = ""
    DB_PORT: int = 5432
    DB_NAME: str = "ai_knowledge_db"
    DB_USER: str = ""
    # 异步模式ORM URL
    ASYNC_DATABASE_URL: str = ""
    # 异步模式Langchain URL
    LANGCHAIN_DATABASE_URL: str = ""

    # 数据库连接池配置
    DB_POOL_MIN_SIZE: int = 1
    DB_POOL_MAX_SIZE: int = 10
    DB_POOL_TIMEOUT: int = 60

    # 嵌入模型的维度，RAG的切块设置
    EMBEDDING_DIM: int = 1536
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CONTEXT_CHUNKS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    VECTOR_SEARCH_LIMIT: int = 10
    KEYWORD_SEARCH_LIMIT: int = 10

    BATCH_SIZE: int = 32
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 1

    # 日志配置
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    LOG_FILE_NAME: str = "logs/server.log"

    # 文件上传配置
    SUPPORTED_EXTENSIONS: list[str] = ['.txt', '.xlsx', '.xls', '.csv']
    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    TEMP_DIR: str = "/tmp/rag_uploads"

    # 服务器地址
    API_HOST: str = "101.132.36.117"
    API_PORT: int = 8000
    API_RELOAD: bool = False

    # 深度搜索
    DEEP_SEARCH_AGENT_ID: str = ""
    DEEP_SEARCH_AGENT_VERSION: str = "beta"

    @property
    def database_params(self) -> Dict[str, Any]:
        return {
            'host': self.DB_HOST,
            'port': self.DB_PORT,
            'database': self.DB_NAME,
            'user': self.DB_USER,
            'password': self.DB_PASSWORD,
            'min_size': self.DB_POOL_MIN_SIZE,
            'max_size': self.DB_POOL_MAX_SIZE,
            'command_timeout': self.DB_POOL_TIMEOUT
        }

# 全局单例配置对象
config = Config()