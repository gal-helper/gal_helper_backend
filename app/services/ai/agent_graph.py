from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph

from app.core.langchain import langchain_manager


async def get_gal_agent() -> CompiledStateGraph:  # -> CompiledStateGraph
    model = await langchain_manager.get_base_chat_model()
    if model is None:
        raise RuntimeError("Chat model not initialized")

    from .agent_tools import tools as gal_tools

    checkpointer = await langchain_manager.get_checkpointer()

    agent = create_agent(
        model=model,
        tools=gal_tools,
        checkpointer=checkpointer,
        system_prompt="你是一个专业的 Galgame 助手，请用亲切的语气回答用户。",
        debug=False,
    )
    return agent
