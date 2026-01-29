from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_info import ChatSession
from app.crud.utils import utils_crud

class ChatSessionCRUD:
    async def create(self, db: AsyncSession, session_code: str) -> str:
        """
        创建一个新的聊天会话
        """
        # 生成主键id
        next_id = await utils_crud.get_next_id(db, ChatSession.__tablename__)
        new_chat_session = ChatSession(
            id=next_id,
            chat_session_code=session_code,
        )
        db.add(new_chat_session)
        return session_code

    async def get_by_code(self, db: AsyncSession, session_code: str) -> ChatSession | None:
        """
        根据session_code获取聊天会话
        """
        stmt = select(ChatSession).where(ChatSession.chat_session_code == session_code)
        result = await db.execute(stmt)
        return result.scalars().first()

# 实例化单例，供其他service模块调用
chat_session_crud = ChatSessionCRUD()

