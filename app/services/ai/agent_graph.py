from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from app.services.ai.prompts import system_prompts
from app.core.langchain import langchain_manager


async def get_gal_agent() -> CompiledStateGraph:  # -> CompiledStateGraph
    """构建一个demo级别的agent"""
    model = await langchain_manager.get_base_chat_model()
    if model is None:
        raise RuntimeError("Chat model not initialized")

    from .agent_tools import tools as gal_tools

    checkpointer = await langchain_manager.get_checkpointer()

    agent = create_agent(
        model=model,
        tools=gal_tools,
        checkpointer=checkpointer,
        system_prompt=system_prompts.gal_helper_system_prompt,
        debug=False,
    )
    return agent
