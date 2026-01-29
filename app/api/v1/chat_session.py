import logging
from app.utils.response import success_response
from fastapi import APIRouter, Depends
from app.services.chat_info import get_chat_session_service, ChatSessionService

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/create")
async def create_chat_session(chat_session_service: ChatSessionService = Depends(get_chat_session_service)):
    """
    创建一个新的聊天会话
    """
    return success_response(await chat_session_service.create())
