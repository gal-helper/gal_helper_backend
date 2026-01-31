from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.chat_info import ChatSession, ChatMessage
from app.crud.utils import utils_crud
from app.utils.constants import ChatRole


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

    async def get_by_session_code(
        self, db: AsyncSession, session_code: str
    ) -> ChatSession | None:
        """
        根据session_code获取聊天会话
        """
        stmt = select(ChatSession).where(ChatSession.chat_session_code == session_code)
        result = await db.execute(stmt)
        return result.scalars().first()

    async def update_message_id_of_session(
        self, db: AsyncSession, session_id: int, message_id: int
    ) -> None:
        """
        根据session_id更新current_message_id为传入的message_id
        """
        # 1.查询对象
        session = await db.get(ChatSession, session_id)
        if session is None:
            raise Exception("会话不存在")
        # 2. 直接修改属性
        session.current_message_id = message_id


# 实例化单例，供其他service模块调用
chat_session_crud = ChatSessionCRUD()


class ChatMessageCRUD:
    """
    聊天消息的CRUD
    """

    async def insert_message(
        self,
        db: AsyncSession,
        chat_session: ChatSession,
        parent_message_id: int,
        message_text: str,
        role: str,
    ) -> int:
        """
        插入一条新的聊天消息
        """
        next_id = await utils_crud.get_next_id(db, ChatMessage.__tablename__)
        message_id = 1 if parent_message_id is None else parent_message_id + 1
        chat_message = ChatMessage(
            id=next_id,
            fk_session_id=chat_session.id,
            message_id=message_id,
            parent_id=parent_message_id,
            role=role,
            message=message_text,
        )
        db.add(chat_message)
        return next_id

    async def insert_ai_message(
        self,
        db: AsyncSession,
        chat_session: ChatSession,
        parent_message_id: int,
        message_text: str,
    ) -> int:
        """
        插入一条新的AI生成的聊天消息
        """
        return await self.insert_message(
            db, chat_session, parent_message_id, message_text, ChatRole.ASSISTANT
        )

    async def insert_user_message(
        self,
        db: AsyncSession,
        chat_session: ChatSession,
        parent_message_id: int,
        message_text: str,
    ) -> int:
        """
        插入一条新的用户输入的聊天消息
        """
        return await self.insert_message(
            db, chat_session, parent_message_id, message_text, ChatRole.USER
        )

    async def update_ai_message(
        self, db: AsyncSession, id: int, message_text: str
    ) -> None:
        """
        根据message表的id更新AI生成的消息内容
        """
        chat_message = await db.get(ChatMessage, id)
        if chat_message is None:
            raise Exception("消息不存在")
        chat_message.message = message_text


# 实例化单例，供其他service模块调用
chat_message_crud = ChatMessageCRUD()
