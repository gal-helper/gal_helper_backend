from . import chat, system
from fastapi import APIRouter
api_router = APIRouter()
api_router.include_router(system.router, tags=["System"]) # 不带 prefix，就是根
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])