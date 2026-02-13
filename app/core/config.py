from typing import Literal
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Config(BaseSettings):
    # ========== 模型配置（OpenAI兼容） ==========
    CHAT_MODEL_BASE_URL: str = ""
    CHAT_MODEL_NAME: str = ""
    CHAT_MODEL_API_KEY: str = ""

    BASE_EMBEDDING_MODEL_BASE_URL: str = ""
    BASE_EMBEDDING_MODEL_NAME: str = ""
    BASE_EMBEDDING_API_KEY: str = ""

    # ========== 数据库配置 ==========
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "ai_knowledge_db"
    DB_USER: str = ""
    DB_PASSWORD: str = ""

    ASYNC_DATABASE_URL: str = ""
    LANGCHAIN_DATABASE_URL: str = ""

    # ========== RAG 参数 ==========
    EMBEDDING_DIM: int = 1536
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_CONTEXT_CHUNKS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7

    # ========== 服务器 ==========
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING"] = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


config = Config()