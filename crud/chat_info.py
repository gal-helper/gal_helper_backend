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
        """更新会话的当前消息ID"""
        session = await db.get(ChatSession, session_id)
        if session:
            session.current_message_id = message_id
            await db.flush()


chat_session_crud = ChatSessionCRUD()


class ChatMessageCRUD:
    async def insert_user_message(self, db: AsyncSession, session_info, current_message_id, ask_text):
        """插入用户消息"""
        # 获取下一个ID
        result = await db.execute(
            select(func.nextval('ai_message_info_id_seq'))
        )
        next_id = result.scalar() or 1

        new_message = ChatMessage(
            id=next_id,
            fk_session_id=session_info.id,
            message_id=current_message_id + 1,  # 用户消息ID
            parent_id=current_message_id,       # 父消息ID（如果没有则为0）
            role="user",
            message=ask_text
        )
        db.add(new_message)
        await db.flush()
        return new_message.id

    async def insert_ai_message(self, db: AsyncSession, session_info, message_id, content):
        """插入AI消息"""
        result = await db.execute(
            select(func.nextval('ai_message_info_id_seq'))
        )
        next_id = result.scalar() or 1

        new_message = ChatMessage(
            id=next_id,
            fk_session_id=session_info.id,
            message_id=message_id + 1,  # AI消息ID（比用户消息大1）
            parent_id=message_id,        # 父消息ID（指向用户消息）
            role="assistant",
            message=content or ""
        )
        db.add(new_message)
        await db.flush()
        return new_message.id

    async def update_message(self, db: AsyncSession, message_id: int, content: str):
        """更新消息内容"""
        stmt = select(ChatMessage).where(ChatMessage.id == message_id)
        result = await db.execute(stmt)
        message = result.scalar_one_or_none()
        if message:
            message.message = content
            await db.flush()

    async def get_all_messages_of_session(self, db: AsyncSession, session_id: int):
        """获取会话的所有消息，按message_id排序"""
        stmt = select(ChatMessage).where(ChatMessage.fk_session_id == session_id).order_by(ChatMessage.message_id)
        result = await db.execute(stmt)
        return result.scalars().all()


chat_message_crud = ChatMessageCRUD()