from langchain_core.tools import tool
from app.core.langchain import langchain_manager
from app.core.config import config
from app.reranker.reranker import reranker
import logging
import json
from typing import Optional

logger = logging.getLogger(__name__)


@tool
async def retrieve_documents(query: str, k: int = 5, table: Optional[str] = None, topic: Optional[str] = None) -> str:
    """
    检索知识库中的相关文档片段，返回 JSON 字符串，包含 items 列表。
    """
    try:
        if table:
            try:
                vectorstore = await langchain_manager.async_get_vectorstore_for_table(table)
            except Exception:
                vectorstore = langchain_manager.get_vectorstore()
                table = None
        else:
            vectorstore = langchain_manager.get_vectorstore()

        initial_k = k * 2 if getattr(config, 'RERANKER_ENABLED', False) else k
        docs = await vectorstore.asimilarity_search(query, k=initial_k)

        if not docs:
            return json.dumps({"items": [], "count": 0}, ensure_ascii=False)

        # 尝试向量余弦重排序（优先于 CrossEncoder）
        if reranker.is_embedding_available() and len(docs) > 1:
            try:
                doc_contents = [doc.page_content for doc in docs]
                reranked = reranker.rerank_by_vector_cosine(query, doc_contents, top_k=k)

                new_docs = []
                used = set()
                for content, score in reranked:
                    for i, doc in enumerate(docs):
                        if i not in used and doc.page_content == content:
                            new_docs.append((doc, score))
                            used.add(i)
                            break

                if len(new_docs) == k:
                    docs = [d for d, _ in new_docs]
                    logger.debug("Vector cosine reranking applied successfully")
            except Exception as e:
                logger.warning(f"Vector cosine reranking failed, trying CrossEncoder: {e}")
                # 降级到 CrossEncoder
                if getattr(config, 'RERANKER_ENABLED', False) and reranker.is_available() and len(docs) > 1:
                    try:
                        doc_contents = [doc.page_content for doc in docs]
                        reranked = reranker.rerank(query, doc_contents, top_k=k)

                        new_docs = []
                        used = set()
                        for content, _ in reranked:
                            for i, doc in enumerate(docs):
                                if i not in used and doc.page_content == content:
                                    new_docs.append(doc)
                                    used.add(i)
                                    break

                        if len(new_docs) == k:
                            docs = new_docs
                            logger.debug("CrossEncoder reranking applied")
                    except Exception as e2:
                        logger.warning(f"CrossEncoder reranking also failed: {e2}")
                        docs = docs[:k]
                else:
                    docs = docs[:k]
        elif getattr(config, 'RERANKER_ENABLED', False) and reranker.is_available() and len(docs) > 1:
            # 仅 CrossEncoder 可用
            try:
                doc_contents = [doc.page_content for doc in docs]
                reranked = reranker.rerank(query, doc_contents, top_k=k)

                new_docs = []
                used = set()
                for content, _ in reranked:
                    for i, doc in enumerate(docs):
                        if i not in used and doc.page_content == content:
                            new_docs.append(doc)
                            used.add(i)
                            break

                if len(new_docs) == k:
                    docs = new_docs
                    logger.debug("CrossEncoder reranking applied successfully")
            except Exception as e:
                logger.warning(f"CrossEncoder reranking failed: {e}")
                docs = docs[:k]
        else:
            docs = docs[:k]

        items = []
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get("filename", "未知文档") if doc.metadata else "未知文档"
            content = doc.page_content.strip()
            metadata = dict(doc.metadata) if doc.metadata else {}
            items.append({
                "index": i,
                "filename": filename,
                "content": content,
                "metadata": metadata,
                "table": table,
                "topic": topic or metadata.get('topic')
            })

        return json.dumps({"items": items, "count": len(items)}, ensure_ascii=False)

    except Exception as e:
        logger.error(f"检索失败: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@tool
async def rewrite_search_query(original_query: str, context: str) -> str:
    return f"{original_query} 相关信息"
from langchain_core.tools import tool
from app.core.langchain import langchain_manager
from app.core.config import config
from app.reranker.reranker import reranker
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
        vectorstore = langchain_manager.get_vectorstore()

        # 如果启用重排序，多取一些文档供排序
        initial_k = k * 2 if config.RERANKER_ENABLED else k
        docs = await vectorstore.asimilarity_search(query, k=initial_k)

        if not docs:
            return "未找到相关文档。"

        # 尝试重排序（如果可用且启用）
        if config.RERANKER_ENABLED and reranker.is_available() and len(docs) > 1:
            try:
                # 提取内容
                doc_contents = [doc.page_content for doc in docs]

                # 重排序
                reranked = reranker.rerank(query, doc_contents, top_k=k)

                # 重建 docs（保持原顺序）
                new_docs = []
                used = set()
                for content, _ in reranked:
                    for i, doc in enumerate(docs):
                        if i not in used and doc.page_content == content:
                            new_docs.append(doc)
                            used.add(i)
                            break

                # 如果重排序成功，用新顺序
                if len(new_docs) == k:
                    docs = new_docs
                    logger.debug("Reranking applied successfully")
            except Exception as e:
                # 重排序失败就继续用原来的
                logger.warning(f"Reranking failed, using original order: {e}")
                docs = docs[:k]
        else:
            docs = docs[:k]

        # 格式化输出（和你原来一模一样）
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
    return f"{original_query} 相关信息"