from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import api_router
from app.core.logging import setup_logging
from app.core.lifespan import lifespan

setup_logging()

# 1. 实例化 FastAPI
app = FastAPI(
    title="AI RAG API",
    description="API for AI RAG Question Answering System",
    version="1.0.0",
    lifespan=lifespan,
)

# 2. 配置中间件，目前有跨域中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 挂载路由
app.include_router(api_router, prefix="/api/v1")
