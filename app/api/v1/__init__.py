from . import chat, chat_session, system, recursive_search
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(system.router, tags=["System"])  # 不带 prefix，就是根
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(chat_session.router, prefix="/chat_session", tags=["chat_session"])
api_router.include_router(recursive_search.router, prefix="/search", tags=["recursive-retrieval"])