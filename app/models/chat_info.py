from datetime import datetime
from typing import Dict, Any

from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import Integer, String

class Base(DeclarativeBase):
    create_time: Mapped[str] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="创建时间"
    )
    update_time: Mapped[str] = mapped_column(
        DateTime,
        default=datetime.now,
        comment="更新时间"
    )

class ChatSession(Base):
    __tablename__ = "ai_chat_session_info"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False, comment="会话表ID")
    chat_session_code: Mapped[str] = mapped_column(String, unique=True, nullable=False, comment="会话编码")
    user_intent: Mapped[int] = mapped_column(Integer, comment="用户意图")
    current_message_id: Mapped[int] = mapped_column(Integer, comment="当前消息ID")

    def __repr__(self):
        return f"<ChatSession(id={self.id}, chat_session_code={self.chat_session_code}, user_intent={self.user_intent}, current_message_id={self.current_message_id})>"

class ChatSessionMemory(Base):

    __tablename__ = "ai_chat_session_memory"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False, comment="会话历史ID")
    chat_session_code: Mapped[str] = mapped_column(String, unique=True, nullable=False, comment="会话编码")
    # 使用jsonb来存储会话历史
    chat_session_memory: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict, comment="会话历史")

    def __repr__(self):
        return f"<ChatSessionMemory(id={self.id}, chat_session_code={self.chat_session_code}, chat_session_memory={self.chat_session_memory})>"