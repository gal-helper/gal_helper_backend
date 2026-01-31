from datetime import datetime
from typing import Dict, Any

from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import Integer, String

class Base(DeclarativeBase):
    create_time: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        comment="创建时间"
    )
    update_time: Mapped[datetime] = mapped_column(
        DateTime,
        default=func.now(),
        onupdate=func.now(),
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

class ChatMessage(Base):
    __tablename__ = "ai_message_info"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False, comment="消息ID")
    fk_session_id: Mapped[int] = mapped_column(Integer, comment="会话ID")
    message_id: Mapped[int] = mapped_column(Integer, comment="在当前会话中的消息序列")
    parent_id: Mapped[int] = mapped_column(Integer, comment="父级消息序列")
    role: Mapped[str] = mapped_column(String, comment="消息角色")
    message: Mapped[str] = mapped_column(String, comment="消息内容")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, fk_session_id={self.fk_session_id}, message_id={self.message_id}, parent_id={self.parent_id}, role={self.role}, message={self.message})>"
