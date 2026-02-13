from datetime import datetime
import logging

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_async_dbsession
from app.models.chat_info import ChatSession, ChatMessage
from app.services.ai.agent_graph import get_gal_agent
from app.utils.utils import UUIDUtil, SSEUtil
from app.crud.chat_info import chat_session_crud, chat_message_crud
from app.utils.constants import EventType, ChatRole
from app.schemas.chat_info import ChatHistoryMessagesResponse

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


async def build_history_message(
    chat_session: ChatSession, chat_messages: list[ChatMessage]
) -> ChatHistoryMessagesResponse:
    from app.schemas.chat_info import ChatSession as ChatSessionSchema
    from app.schemas.chat_info import ChatMessage as ChatMessageSchema

    chat_session_schema = ChatSessionSchema(
        id=chat_session.chat_session_code,
        updated_at=chat_session.update_time,
        version=chat_session.current_message_id,
        current_message_id=chat_session.current_message_id,
        inserted_at=chat_session.create_time,
    )

    chat_messages_schema = [
        ChatMessageSchema(
            message_id=msg.message_id,
            parent_id=msg.parent_id,
            role=ChatRole(msg.role),
            message_content=msg.message,
            accumulated_token_usage=0,
            inserted_at=msg.create_time,
        )
        for msg in chat_messages
    ]

    return ChatHistoryMessagesResponse(
        chat_session=chat_session_schema, chat_messages=chat_messages_schema
    )


class ChatMessageService:
    def __init__(self, db: AsyncSession, agent: CompiledStateGraph):
        self.db = db
        self.agent = agent

    async def get_history_message(self, session_code: str):
        logger.info(f"获取会话编码为：{session_code}的历史消息")
        # 1. 查询获取会话信息
        chat_session = await chat_session_crud.get_by_session_code(
            self.db, session_code
        )
        if not chat_session:
            logger.error(f"会话编码为：{session_code}的会话不存在")
            raise Exception("会话不存在")
        # 2. 根据会话id查询所有消息
        chat_messages: list = await chat_message_crud.get_all_messages_of_session(
            self.db, chat_session.id
        )
        if not chat_messages:
            logger.info(f"会话编码为：{session_code}的会话没有历史消息")
            chat_messages = []
        # 3. 构建历史消息响应数据
        return await build_history_message(chat_session, chat_messages)

    async def chat(self, session_code: str, ask_text: str):
        # --- 事件 1 ready ---
        chat_session_info: ChatSession = await chat_session_crud.get_by_session_code(
            self.db, session_code
        )
        if not chat_session_info:
            raise Exception("会话不存在")
        logger.info(
            f"会话信息编码:{chat_session_info.chat_session_code}，当前的会话序列:{chat_session_info.current_message_id}，"
            f"用户的问题为：{ask_text}"
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

        # --- 事件 3 AI 内容流 ---
        full_response = ""

        # 构建 Agent 输入
        inputs = {
            "input": ask_text,
            "chat_history": await self._build_chat_history(session_code),
        }

        # 流式执行 Agent
        async for event in self.agent.astream_events(
                inputs,
                version="v2",
                config={"configurable": {"thread_id": session_code}}
        ):
            kind = event["event"]

            # 工具调用事件 - 推给前端显示
            if kind == "on_tool_start":
                yield SSEUtil.format_sse(
                    event=EventType.REASONING,
                    data={"tool": event["name"], "status": "start"}
                )
            elif kind == "on_tool_end":
                yield SSEUtil.format_sse(
                    event=EventType.REASONING,
                    data={"tool": event["name"], "status": "end"}
                )

            # LLM 输出流
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    full_response += chunk.content
                    yield SSEUtil.format_sse(
                        event=EventType.MESSAGE,
                        data={"content": chunk.content}
                    )

        # --- 事件 4 finish ---
        yield SSEUtil.format_sse(event=EventType.FINISH, data={})

        logger.info(f"AI回复为：{full_response}")

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
