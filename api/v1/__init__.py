from . import chat, chat_session
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(chat_session.router, prefix="/chat_session", tags=["chat_session"])
