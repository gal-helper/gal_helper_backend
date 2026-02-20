from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import api_router
from app.core.logging import setup_logging
from app.core.lifespan import lifespan
from scalar_fastapi import get_scalar_api_reference

from app.utils.exception_handlers import register_exception_handlers

setup_logging()

# 1. 实例化 FastAPI
app = FastAPI(
    title="AI RAG API",
    description="API for AI RAG Question Answering System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)
# 注册全局异常处理
register_exception_handlers(app)

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


# 使用scalar作为docs，能好看点
@app.get("/docs", include_in_schema=False)
async def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="API Docs",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="API Docs",
    )
