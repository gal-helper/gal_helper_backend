import logging

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_async_dbsession
from app.utils.utils import UUIDUtil
from app.crud.chat_info import chat_session_crud

logger = logging.getLogger(__name__)

class ChatSessionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(self):
        logger.info("创建一个新的会话")
        session_code = UUIDUtil.generate_v7()
        await chat_session_crud.create(self.db, session_code)
        logger.info(f"创建成功，会话编码为：{session_code}")
        return session_code

# 注入工厂
def get_chat_session_service(db: AsyncSession = Depends(get_async_dbsession)) -> ChatSessionService:
    return ChatSessionService(db)

class ChatMessageService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def chat(self, session_code: str, message: str):

        # --- 事件 1 ready ---
        # 获取用户chat的request_message_id和response_message_id

        # --- 事件 2 update_session ---
        # 存储user_message，并更新会话的message_id

        # --- 事件 3 AI 内容流 (message) ---
        # 这里返回agent的内容流

        # --- 事件 4 finish ---
        # 声明内容流传输结束

        # --- 事件 5 update_session ---
        # 保存agent的生成的ai_message

        # --- 事件 6 close ---

        pass

