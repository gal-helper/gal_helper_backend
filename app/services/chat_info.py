from datetime import datetime
import logging

from fastapi import Depends
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_db
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
        db: AsyncSession = Depends(get_db),
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

    async def chat(self, session_code: str, ask_text: str, topic: str = None):
        # --- 事件 1 ready ---
        chat_session_info: ChatSession = await chat_session_crud.get_by_session_code(
            self.db, session_code
        )
        if not chat_session_info:
            # 如果会话不存在，创建新会话（作为后备）
            logger.warning(f"会话不存在，创建新会话: {session_code}")
            await chat_session_crud.create(self.db, session_code)
            chat_session_info = await chat_session_crud.get_by_session_code(
                self.db, session_code
            )

        # 获取当前消息ID
        current_message_id: int = chat_session_info.current_message_id or 0
        next_message_id: int = current_message_id + 1

        logger.info(
            f"会话信息编码:{chat_session_info.chat_session_code}，当前消息ID:{current_message_id}，下一个消息ID:{next_message_id}，"
            f"用户的问题为：{ask_text}"
        )

        # 获取历史消息（用于日志）
        all_messages = await chat_message_crud.get_all_messages_of_session(
            self.db, chat_session_info.id
        )

        if all_messages:
            logger.info(f"找到 {len(all_messages)} 条历史消息")
            for msg in all_messages[-3:]:  # 只显示最近3条
                logger.debug(f"历史 - {msg.role}: {msg.message[:50]}...")
        else:
            logger.info("没有历史消息，这是新会话的第一条消息")

        yield SSEUtil.format_sse(
            event=EventType.READY,
            data={
                "request_message_id": next_message_id,
                "response_message_id": next_message_id + 1,
            },
        )

        # --- 事件 2 update_session ---
        # 插入用户消息
        user_message_id = await chat_message_crud.insert_user_message(
            self.db, chat_session_info, current_message_id, ask_text
        )
        logger.info(f"插入用户消息成功，ID: {user_message_id}")

        # 插入AI消息（但没有消息内容）
        ai_message_id: int = await chat_message_crud.insert_ai_message(
            self.db, chat_session_info, next_message_id, None
        )
        logger.info(f"插入AI消息成功，ID: {ai_message_id}")

        # 更新会话信息到最新的消息ID
        await chat_session_crud.update_message_id(
            self.db, chat_session_info.id, next_message_id + 1
        )

        yield SSEUtil.format_sse(
            event=EventType.UPDATE_SESSION, data={"updated_at": datetime.now()}
        )

        # --- 事件 3 AI 内容流 ---
        full_response = ""

        # 构建 Agent 输入，包含历史消息
        # 获取当前会话的所有历史消息
        all_messages = await chat_message_crud.get_all_messages_of_session(
            self.db, chat_session_info.id
        )

        # 转换为 LangChain 消息格式
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        messages = []
        for msg in all_messages:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.message))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.message))
            elif msg.role == "system":
                messages.append(SystemMessage(content=msg.message))

        # 添加当前用户消息（确保包含）
        messages.append(HumanMessage(content=ask_text))

        logger.info(f"向Agent发送 {len(messages)} 条消息")

        inputs = {
            "messages": messages,
            "topic": topic,
        }

        # 流式执行 Agent - 使用 updates 模式更容易处理工具调用
        async for chunk in self.agent.astream(
                inputs,
                config={"configurable": {"thread_id": session_code}},
                stream_mode="updates"
        ):
            # 处理不同类型的更新
            for node_name, node_data in chunk.items():
                # 处理消息输出
                if "messages" in node_data:
                    msgs = node_data["messages"]
                    if msgs:
                        # 获取最新的消息
                        last_message = msgs[-1]
                        if hasattr(last_message, "content") and last_message.content:
                            # 只输出新增的内容
                            new_content = last_message.content[len(full_response):]
                            if new_content:
                                full_response = last_message.content
                                yield SSEUtil.format_sse(
                                    event=EventType.MESSAGE,
                                    data={"content": new_content}
                                )

                # 处理工具调用
                if "tools" in node_data:
                    for tool_call in node_data["tools"]:
                        # 如果工具返回了结果（如 retrieve_documents 返回 JSON），则尝试解析并发送 retrieval 事件
                        try:
                            tool_name = tool_call.get("name")
                            result = tool_call.get("result") or tool_call.get("output")
                            if tool_name == "retrieve_documents" and result:
                                import json as _json
                                try:
                                    parsed = _json.loads(result)
                                    items = parsed.get("items") if isinstance(parsed, dict) else None
                                    if items:
                                        for item in items:
                                            yield SSEUtil.format_sse(
                                                event=EventType.RETRIEVAL,
                                                data=item,
                                            )
                                        continue
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        yield SSEUtil.format_sse(
                            event=EventType.REASONING,
                            data={"tool": tool_call.get("name"), "status": "calling"}
                        )

        # --- 事件 4 finish ---
        yield SSEUtil.format_sse(event=EventType.FINISH, data={})

        logger.info(f"AI回复为：{full_response[:100]}...")

        # --- 事件 5 update_session ---
        await chat_message_crud.update_message(self.db, ai_message_id, full_response)
        await self.db.commit()
        yield SSEUtil.format_sse(
            event=EventType.UPDATE_SESSION, data={"updated_at": datetime.now()}
        )

        # --- 事件 6 close ---
        yield SSEUtil.format_sse(event=EventType.CLOSE, data={"click_behavior": "none"})


# 注入工厂
def get_chat_message_service(
        db: AsyncSession = Depends(get_db),
        agent: CompiledStateGraph = Depends(get_gal_agent),
) -> ChatMessageService:
    return ChatMessageService(db, agent)