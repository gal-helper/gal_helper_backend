from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate
from langgraph.graph.state import CompiledStateGraph
from app.core.langchain import langchain_manager
from app.services.agent.galgame_tools import tools as galgame_tools
from app.services.agent.rag_tools import retrieve_documents, rewrite_search_query


async def get_gal_agent() -> CompiledStateGraph:
    """构建完整的 Galgame 助手 Agent，带 RAG 能力"""

    model = await langchain_manager.get_base_chat_model()
    checkpointer = await langchain_manager.get_checkpointer()

    tools = galgame_tools + [retrieve_documents, rewrite_search_query]

    prompt = PromptTemplate.from_template("""你是一个专业的 Galgame 从业者，用简明易懂的语气回答用户问题。
你有以下工具可用：
{tools}
重要规则：
1. 当用户询问具体游戏信息、攻略、问题解决方案时，**必须先用 retrieve_documents 搜索知识库**
2. 如果第一次搜索结果不充分，调用 rewrite_search_query 优化搜索词，然后再次检索
3. 每次引用文档内容，请在句子末尾标注 [数字]，例如 "根据攻略，这里需要转区运行[1]"
4. 最终回答必须基于检索到的文档，不要编造信息

工具名称: {tool_names}
当前对话: {chat_history}
用户输入: {input}
{agent_scratchpad}
部分对话模式如下：""")

    agent = create_react_agent(
        llm=model,
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer,
        max_iterations=5,
        debug=False
    )

    return agent