from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langgraph.graph.state import CompiledStateGraph
from app.core.langchain import langchain_manager
from app.services.ai.agent_tools import tools
from app.services.agent.rag_tools import retrieve_documents, rewrite_search_query
import logging

logger = logging.getLogger(__name__)


def get_gal_agent() -> CompiledStateGraph:
    """构建完整的 Galgame 助手 Agent，带 RAG 能力"""

    model = langchain_manager.get_chat_model()

    # 尝试获取 checkpointer，如果没有就传 None
    try:
        checkpointer = langchain_manager.get_checkpointer()
    except:
        checkpointer = None
        logger.warning("No checkpointer available, continuing without it")

    all_tools = tools + [retrieve_documents, rewrite_search_query]

    # 创建自定义的 prompt 处理函数 - 增强上下文理解
    def prompt_func(state):
        messages = state["messages"]

        # 格式化消息历史，添加上下文标记
        history = []
        for i, msg in enumerate(messages):
            # 根据消息类型确定角色
            msg_type = msg.__class__.__name__
            if msg_type == "HumanMessage":
                role = "用户"
                emoji = "👤"
            elif msg_type == "AIMessage":
                role = "AI助手"
                emoji = "🤖"
            elif msg_type == "SystemMessage":
                role = "系统"
                emoji = "⚙️"
            else:
                role = "未知"
                emoji = "❓"

            # 添加上下文标记，帮助AI理解对话顺序
            if i == len(messages) - 1 and role == "用户":
                context = "【当前问题】"
            elif i == 0:
                context = "【对话开始】"
            else:
                context = f"【第{i + 1}轮】"

            history.append(f"{context} {emoji} {role}: {msg.content}")

        # 获取最新的用户消息作为输入
        user_input = messages[-1].content if messages and hasattr(messages[-1], 'content') else ""

        # 获取中间步骤（如果有）
        remaining_steps = state.get("remaining_steps", "")
        steps_text = f"\n\n🔧 中间步骤：\n{remaining_steps}" if remaining_steps else ""

        # 判断是否有历史对话
        has_history = len(messages) > 1

        # 构建完整的提示词
        prompt = f"""你是一个专业的 Galgame 从业者，用简明易懂的语气回答用户问题。

📋 **重要规则：**
1. 当用户询问具体游戏信息、攻略、问题解决方案时，**必须先用 retrieve_documents 搜索知识库**
2. 如果第一次搜索结果不充分，调用 rewrite_search_query 优化搜索词，然后再次检索
3. 每次引用文档内容，请在句子末尾标注 [数字]，例如 "根据攻略，这里需要转区运行[1]"
4. 最终回答必须基于检索到的文档，不要编造信息
5. **仔细阅读对话历史，理解上下文** - 用户可能会问"为什么"、"然后呢"这样的后续问题

{"💬 **对话历史：**" if has_history else "🆕 **新对话开始**"}
{chr(10).join(history) if history else "暂无历史对话"}

❓ **用户最新问题：**
{user_input}{steps_text}

请基于以上对话历史回答用户问题。如果是后续问题（如"为什么"、"然后呢"），请结合之前的对话内容回答。"""

        return prompt

    # 使用 LangGraph 的 create_react_agent
    agent = create_react_agent(
        model=model,
        tools=all_tools,
        prompt=prompt_func,
        checkpointer=checkpointer,
    )

    return agent