import json
from datetime import datetime, date
from uuid import UUID
import uuid6


class JSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，支持datetime、UUID等"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


class UUIDUtil:
    """
    UUID工具类
    """

    @staticmethod
    def generate_v7() -> UUID:
        """
        生成UUIDv7
        - 前 48 位是毫秒级时间戳
        - 后面是随机数
        - 优点：按时间排序，数据库索引友好
        """
        return uuid6.uuid7()

    @staticmethod
    def to_str(uid: UUID) -> str:
        return str(uid)


class SSEUtil:
    """
    SSE工具类
    """

    @staticmethod
    def format_sse(data: dict, event: str = None) -> str:
        """辅助函数：将字典转换为 SSE 格式字符串"""
        msg = f"data: {json.dumps(data, cls=JSONEncoder)}\n"
        if event:
            msg = f"event: {event}\n{msg}"
        return msg + "\n"
