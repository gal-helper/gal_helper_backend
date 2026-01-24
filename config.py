import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Config:
    
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    EMBEDDING_MODEL = "text-embedding-v2"
    CHAT_MODEL = "qwen-max"
    API_TIMEOUT = 60
    EMBEDDING_TIMEOUT = 30
    
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = 5432
    DB_NAME = "ai_knowledge_db"
    DB_USER = "dick2416910961"
    
    DB_POOL_MIN_SIZE = 1
    DB_POOL_MAX_SIZE = 10
    DB_POOL_TIMEOUT = 60
    
    EMBEDDING_DIM = 1536
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    MAX_CONTEXT_CHUNKS = 5
    SIMILARITY_THRESHOLD = 0.7
    VECTOR_SEARCH_LIMIT = 10
    KEYWORD_SEARCH_LIMIT = 10
    
    BATCH_SIZE = 32
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    LOG_LEVEL = "INFO"
    SUPPORTED_EXTENSIONS = ['.txt', '.xlsx', '.xls', '.csv']
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    TEMP_DIR = "/tmp/rag_uploads"
    
    API_HOST = "101.132.36.117"
    API_PORT = 8000
    API_RELOAD = False
    CORS_ORIGINS = ["*"]
    
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

config = Config()
