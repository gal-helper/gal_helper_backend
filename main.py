from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api.v1 import api_router
from app.core.logging import setup_logging
from app.core.lifespan import lifespan

from app.utils.exception_handlers import register_exception_handlers

setup_logging()

# 1. 实例化 FastAPI
app = FastAPI(
    title="AI RAG API",
    description="API for AI RAG Question Answering System",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 注册全局异常处理
register_exception_handlers(app)

# 2. 配置中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 挂载路由
app.include_router(api_router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Gal Helper API</title>
        </head>
        <body>
            <h1>Gal Helper API 运行中</h1>
            <p>访问以下地址查看 API 文档：</p>
            <ul>
                <li><a href="/docs">/docs - Swagger UI</a></li>
                <li><a href="/redoc">/redoc - ReDoc</a></li>
            </ul>
        </body>
    </html>
    """


@app.get("/docs")
async def swagger_ui():
    from fastapi.openapi.docs import get_swagger_ui_html
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="API Docs",
    )
