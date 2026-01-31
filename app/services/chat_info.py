from datetime import datetime
import logging

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.db import async_db_manager
from app.core.dependencies import get_async_dbsession
from app.models.chat_info import ChatSession
from app.services.ai.agent_graph import get_gal_agent
from app.utils.utils import UUIDUtil, SSEUtil
from app.crud.chat_info import chat_session_crud, chat_message_crud
from app.utils.constants import EventType, ChatRole

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
def get_chat_session_service(
    db: AsyncSession = Depends(get_async_dbsession),
) -> ChatSessionService:
    return ChatSessionService(db)


class ChatMessageService:
    def __init__(self, db: AsyncSession, agent: CompiledStateGraph):
        self.db = db
        self.agent = agent

    async def chat(self, session_code: str, ask_text: str):
        # --- 事件 1 ready ---
        chat_session_info: ChatSession = await chat_session_crud.get_by_session_code(
            self.db, session_code
        )
        if not chat_session_info:
            raise Exception("会话不存在")
        logger.info(
            f"会话信息编码:{chat_session_info.chat_session_code}，当前的会话序列:{chat_session_info.current_message_id}"
        )
        current_message_id: int = chat_session_info.current_message_id
        next_message_id: int = (
            1 if current_message_id is None else current_message_id + 1
        )

        yield SSEUtil.format_sse(
            event=EventType.READY,
            data={
                "request_message_id": next_message_id,
                "response_message_id": next_message_id + 1,
            },
        )

        # --- 事件 2 update_session ---
        # 插入用户消息
        await chat_message_crud.insert_user_message(
            self.db, chat_session_info, current_message_id, ask_text
        )
        # 插入AI消息（但没有消息内容）
        ai_message_id: int = await chat_message_crud.insert_ai_message(
            self.db, chat_session_info, next_message_id, None
        )
        # 更新会话信息
        await chat_session_crud.update_message_id_of_session(
            self.db, chat_session_info.id, next_message_id + 1
        )

        yield SSEUtil.format_sse(
            event=EventType.UPDATE_SESSION, data={"updated_at": datetime.now()}
        )

        # --- 事件 3 AI 内容流 (message) ---
        full_response = ""
        async for chunk in self.agent.astream(
            {"messages": [{"role": ChatRole.USER, "content": ask_text}]},
            stream_mode="messages",
            config={"configurable": {"thread_id": session_code}},
        ):
            token, metadata = chunk
            if hasattr(token, "content") and token.content:
                full_response += token.content
                yield SSEUtil.format_sse(
                    event=EventType.MESSAGE,
                    data={"content": token.content},
                )

        # --- 事件 4 finish ---
        yield SSEUtil.format_sse(event=EventType.FINISH, data={})

        # --- 事件 5 update_session ---
        await chat_message_crud.update_ai_message(self.db, ai_message_id, full_response)
        await self.db.commit()
        yield SSEUtil.format_sse(
            event=EventType.UPDATE_SESSION, data={"updated_at": datetime.now()}
        )

        # --- 事件 6 close ---
        yield SSEUtil.format_sse(event=EventType.CLOSE, data={"click_behavior": "none"})


# 注入工厂
async def get_chat_message_service(
    db: AsyncSession = Depends(get_async_dbsession),
    agent: CompiledStateGraph = Depends(get_gal_agent),
) -> ChatMessageService:
    return ChatMessageService(db, agent)
