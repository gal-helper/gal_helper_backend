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
        # 获取向量存储
        if table:
            try:
                vectorstore = await langchain_manager.async_get_vectorstore_for_table(table)
            except Exception as e:
                logger.debug(f"Failed to get vectorstore for table '{table}': {e}")
                vectorstore = langchain_manager.get_vectorstore()
                table = None
        else:
            vectorstore = langchain_manager.get_vectorstore()
        
        # 🛡️ 检查向量存储是否可用
        if vectorstore is None:
            logger.error("❌ Vectorstore is None - ai_documents table not available or not initialized")
            return json.dumps({
                "items": [], 
                "count": 0,
                "error": "Knowledge base not initialized. Please import documents first."
            }, ensure_ascii=False)
        
        # 执行向量相似度搜索
        try:
            # 统一使用 getattr 确保一致性
            initial_k = k * 2 if getattr(config, 'RERANKER_ENABLED', False) else k
            docs = await vectorstore.asimilarity_search(query, k=initial_k)
        except AttributeError as e:
            logger.error(f"❌ Vectorstore method error: {e} - vectorstore type: {type(vectorstore)}")
            return json.dumps({
                "items": [],
                "count": 0,
                "error": "Vectorstore interface error"
            }, ensure_ascii=False)

        if not docs:
            logger.debug(f"No documents found for query: {query}")
            return json.dumps({"items": [], "count": 0}, ensure_ascii=False)

        # 尝试向量余弦重排序（优先于 CrossEncoder）
        if getattr(config, 'RERANKER_ENABLED', False) and reranker.is_embedding_available() and len(docs) > 1:
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
                    logger.debug("CrossEncoder reranking applied")
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
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"❌ Document retrieval failed: {error_msg}")
        
        # 提供友好的错误信息
        if "NoneType" in str(type(e)):
            friendly_error = "Knowledge base not initialized. Please import documents first."
        elif "asimilarity_search" in error_msg:
            friendly_error = "Vectorstore interface error - knowledge base may not be properly initialized"
        else:
            friendly_error = f"Retrieval error: {str(e)}"
        
        return json.dumps({
            "items": [], 
            "count": 0,
            "error": friendly_error
        }, ensure_ascii=False)


@tool
async def rewrite_search_query(original_query: str, context: str) -> str:
    """
    重写搜索查询以改进搜索结果。
    
    Args:
        original_query: 原始查询
        context: 上下文信息
        
    Returns:
        重写后的查询字符串
    """
    return f"{original_query} 相关信息"