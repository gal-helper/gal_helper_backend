import os
import json
import tempfile
import uuid6

from fastapi.responses import StreamingResponse
from fastapi import APIRouter, UploadFile, HTTPException, Depends
from fastapi.params import Form, File
import logging

from app.services.chat_info import ChatMessageService, get_chat_message_service
from app.utils.response import success_response
from app.core.langchain import langchain_manager
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from langchain_core.documents import Document


def _map_topic_to_table(topic: str) -> str:
    """将用户友好主题映射为数据库表名；如果传入已是表名则原样返回。"""
    if not topic:
        return None
    t = topic.strip()
    # 常见表名直接返回
    known_tables = [
        "vectorstore_resource",
        "vectorstore_technical",
        "vectorstore_tools",
        "vectorstore_news",
    ]
    if t in known_tables:
        return t

    # 简单关键字映射（支持中文或英文关键词）
    if any(k in t for k in ["资源", "resource"]):
        return "vectorstore_resource"
    if any(k in t for k in ["运行", "运行问题", "technical", "运行问题", "运行"]):
        return "vectorstore_technical"
    if any(k in t for k in ["工具", "软件", "tool", "tools"]):
        return "vectorstore_tools"
    if any(k in t for k in ["资讯", "新闻", "news", "游戏资讯"]):
        return "vectorstore_news"

    # 默认返回 None，使用默认 vectorstore
    return None

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/ask")
async def ask_question(
        question: str = Form(...),
        use_rag: bool = Form(True),  # 保留参数，但不使用
    chat_message_service: ChatMessageService = Depends(get_chat_message_service),
    topic: str = Form(None),
):
    """
    兼容旧接口：转发到新版的 /completion 流式接口
    返回格式保持与旧版一致（非流式）
    """
    try:
        session_code = f"temp_{uuid6.uuid7()}"

        # 收集流式响应的完整内容
        full_answer = ""
        sources = []

        async for chunk in chat_message_service.chat(session_code, question, topic=_map_topic_to_table(topic) or topic):
            # 解析 SSE 格式
            if chunk.startswith("data: "):
                try:
                    data = json.loads(chunk[6:])
                    if data.get("event") == "message":
                        full_answer += data["data"]["content"]
                    elif data.get("event") == "retrieval":
                        sources.append(data["data"])
                except:
                    pass

        return {
            "success": True,
            "question": question,
            "answer": full_answer,
            "sources": sources,
            "rag_used": True,
            "response_time": 0,
            "error": None
        }

    except Exception as e:
        logger.error(f"/ask 转发失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
    vectorstore: AsyncPGVectorStore = Depends(lambda: langchain_manager.get_vectorstore()),
    topic: str = Form(None),
):
    tmp_path = None
    try:
        # 保存上传的文件到临时路径
        content = await file.read()
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # 简单的文本处理，实际应该用 TextProcessor
        text_content = content.decode('utf-8')

        # 简单分块
        chunk_size = 1000
        chunks = []
        for i in range(0, len(text_content), chunk_size):
            chunks.append(text_content[i:i + chunk_size])

        docs = []
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "filename": file.filename,
                        "file_type": file.content_type,
                        "chunk_index": i
                    }
                )
            )

        # 如果传入了 topic（或 table 名），尝试使用该表保存
        table_name = _map_topic_to_table(topic)
        if table_name:
            try:
                vs = await langchain_manager.async_get_vectorstore_for_table(table_name)
                ids = await vs.aadd_documents(docs)
            except Exception:
                ids = await vectorstore.aadd_documents(docs)
        else:
            ids = await vectorstore.aadd_documents(docs)

        return {
            "success": True,
            "filename": file.filename,
            "document_ids": ids,
            "chunks": len(ids)
        }

    except Exception as e:
        logger.error(f"上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.post("/completion")
async def completion(
        chatSessionCode: str,
        askText: str,
        chat_message_service: ChatMessageService = Depends(get_chat_message_service),
        topic: str = None,
):
    return StreamingResponse(
        chat_message_service.chat(chatSessionCode, askText, topic=_map_topic_to_table(topic) or topic),
        media_type="text/event-stream",
    )


@router.get("/history_messages")
async def get_history_messages(
        chatSessionCode: str,
        chat_message_service: ChatMessageService = Depends(get_chat_message_service),
):
    return success_response(
        "获取历史消息成功",
        await chat_message_service.get_history_message(chatSessionCode),
    )