# app/services/agent_tools.py
from langchain_core.tools import tool

@tool
async def search_galgame_info(game_name: str):
    """当用户询问特定 Galgame 的资讯、发行日期或基本信息时使用。"""
    # 这里的逻辑可以去调你自己的 CRUD 或外部 API
    return f"{game_name} 是一款非常经典的作，发行于..."

@tool
async def solve_game_error(error_log: str):
    """当用户提供游戏运行报错、DLL 缺失或乱码问题时使用。"""
    return "建议尝试使用 Locale Emulator 挂载日区运行..."

# 导出工具列表
tools = [search_galgame_info, solve_game_error]