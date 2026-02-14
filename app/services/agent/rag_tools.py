from langchain_core.tools import tool
from app.core.langchain import langchain_manager
import logging

logger = logging.getLogger(__name__)


@tool
async def retrieve_documents(query: str, k: int = 5) -> str:
    """
    检索知识库中的相关文档片段。
    当用户询问上传文档中的具体内容时使用。

    Args:
        query: 用户的搜索关键词或问题
        k: 返回的文档片段数量，默认5个

    Returns:
        格式化的文档内容字符串
    """
    try:
        vectorstore = await langchain_manager.get_vectorstore()

        # 执行相似度搜索
        docs = await vectorstore.asimilarity_search(
            query,
            k=k,
        )

        if not docs:
            return "未找到相关文档。"

        # 格式化输出
        result = []
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("filename", "未知文档")
            content = doc.page_content.strip()
            result.append(f"[{i}] 来自《{filename}》:\n{content[:500]}...")

        return "\n\n".join(result)

    except Exception as e:
        logger.error(f"检索失败: {e}")
        return f"检索过程出现错误: {str(e)}"


@tool
async def rewrite_search_query(original_query: str, context: str) -> str:
    """
    根据已有的检索结果，重写搜索词以获得更精准的信息。

    Args:
        original_query: 用户的原始问题
        context: 已检索到的文档片段

    Returns:
        优化后的搜索词
    """
    # 简化版本，直接返回原查询加上提示
    return f"{original_query} 相关信息"