import os
from dotenv import load_dotenv

load_dotenv()


class Config:

    # 数据库配置（从 .env 读取）
    DB_HOST: str = os.getenv("DB_HOST")
    DB_PORT: int = int(os.getenv("DB_PORT"))
    DB_USER: str = os.getenv("DB_USER")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD")
    DB_NAME: str = os.getenv("DB_NAME")

    # 完整的数据库 URL（从 .env 读取）
    ASYNC_DATABASE_URL: str = os.getenv("ASYNC_DATABASE_URL")
    LANGCHAIN_DATABASE_URL: str = os.getenv("LANGCHAIN_DATABASE_URL")

    # 模型配置（从 .env 读取）
    CHAT_MODEL_NAME: str = os.getenv("CHAT_MODEL_NAME")
    CHAT_MODEL_BASE_URL: str = os.getenv("CHAT_MODEL_BASE_URL")
    CHAT_MODEL_API_KEY: str = os.getenv("CHAT_MODEL_API_KEY")

    BASE_EMBEDDING_MODEL_NAME: str = os.getenv("BASE_EMBEDDING_MODEL_NAME")
    BASE_EMBEDDING_MODEL_BASE_URL: str = os.getenv("BASE_EMBEDDING_MODEL_BASE_URL")
    BASE_EMBEDDING_API_KEY: str = os.getenv("BASE_EMBEDDING_API_KEY")

    DB_POOL_MIN_SIZE: int = 2
    DB_POOL_MAX_SIZE: int = 10
    VECTOR_DIMENSION: int = 768
    VECTOR_INDEX_LISTS: int = 100
    # 索引创建控制
    AUTO_CREATE_INDEX: bool = True  # 保持 True，让向量索引创建
    SKIP_INDEX_CREATION: bool = False  # 保持 False
    LOG_LEVEL: str = "DEBUG"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_NAME: str = "logs/app.log"
    DEBUG: bool = True


config = Config()

if config.DEBUG:
    print(f"📊 Database: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
    print(f"🤖 Chat Model: {config.CHAT_MODEL_NAME}")
    print(f"🔤 Embedding Model: {config.BASE_EMBEDDING_MODEL_NAME}")
    print(f"🔗 ASYNC_DATABASE_URL: {config.ASYNC_DATABASE_URL}")
    print(f"🔗 LANGCHAIN_DATABASE_URL: {config.LANGCHAIN_DATABASE_URL}")