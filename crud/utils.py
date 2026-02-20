from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
import logging
logger = logging.getLogger(__name__)

class UtilsCRUD:

    async def get_next_id(self, db: AsyncSession, table_name: str):
        """
        获取下一个自增ID
        """
        stmt = select(func.get_next_id(table_name))
        result = await db.execute(stmt)

        next_id = result.scalar()
        logger.info(f"表名：{table_name} 获取下一个自增ID: {next_id}")
        return next_id

utils_crud = UtilsCRUD()