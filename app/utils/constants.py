from enum import Enum

class ChatRole(str, Enum):
    """
    聊天角色枚举
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class EventType(str, Enum):
    """
    事件类型枚举
    """
    READY = "ready"
    UPDATE_SESSION = "update_session"
    MESSAGE = "message"
    FINISH = "finish"
    CLOSE = "close"