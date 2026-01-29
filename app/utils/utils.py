import uuid6
from uuid import UUID

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