from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.chat_info import ChatSession, ChatMessage
import uuid6


class ChatSessionCRUD:
    async def create(self, db: AsyncSession, session_code: str = None) -> str:
        """创建会话"""
        if not session_code:
            session_code = str(uuid6.uuid7())

        # 使用原生 SQL 获取 nextval
        result = await db.execute(
            select(func.nextval(
                'ai_chat_session_info_id_seq' if db.bind.dialect.name == 'postgresql' else 'sqlite_sequence'))
        )
        next_id = result.scalar() or 1

        new_session = ChatSession(
            id=next_id,
            chat_session_code=session_code,
            user_intent=0,  # 默认值
            current_message_id=0
        )
        db.add(new_session)
        await db.flush()  # 不 commit，让上层决定
        return session_code

    async def get_by_session_code(self, db: AsyncSession, session_code: str) -> ChatSession | None:
        stmt = select(ChatSession).where(ChatSession.chat_session_code == session_code)
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_message_id(self, db: AsyncSession, session_id: int, message_id: int):
        session = await db.get(ChatSession, session_id)
        if session:
            session.current_message_id = message_id
            await db.flush()


chat_session_crud = ChatSessionCRUD()