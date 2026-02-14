from typing import List, Optional, Any, Union
from app.utils.constants import ChatRole
from pydantic import BaseModel, Field
from app.utils.constants import ChatRole
from datetime import datetime


class ChatMessage(BaseModel):
    message_id: int
    parent_id: Optional[int] = None
    role: ChatRole
    accumulated_token_usage: int = 0
    message_content: str = Field(default="")
    inserted_at: datetime  # 使用时间戳

class ChatSession(BaseModel):
    id: str
    updated_at: datetime
    version: int
    current_message_id: int
    inserted_at: datetime

# --- 根返回结构 ---

class ChatHistoryMessagesResponse(BaseModel):
    chat_session: ChatSession
    chat_messages: List[ChatMessage]

    class Config:
        # 允许从 SQLAlchemy 模型直接转换
        from_attributes = True