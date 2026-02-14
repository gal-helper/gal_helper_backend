import asyncio
import sys
import os
import argparse

# Windows 专用：设置事件循环策略（必须放在最前面！）
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.chat_info import ChatMessageService
from app.core.langchain import langchain_manager
from app.core.db import async_db_manager, langchain_pool, db_initializer
from app.crud.chat_info import chat_session_crud
from app.services.ai.agent_graph import get_gal_agent
import uuid6
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)


class CLIClient:

    def __init__(self):
        self.db = None
        self.agent = None
        self.chat_service = None
        self.current_session_code = None  # 保存当前会话，实现连续对话

    async def initialize(self) -> bool:
        print("Initializing AI RAG system...")
        try:
            # 1. 初始化数据库连接池
            print("  📦 Connecting to database...")
            await async_db_manager.init_async_database()
            await langchain_pool.connect()

            # 2. 初始化数据库（创建表和索引）
            print("  🗄️ Initializing database schema...")
            await db_initializer.initialize()

            # 3. 获取数据库会话
            print("  📝 Getting database session...")
            async with async_db_manager.get_async_db() as session:
                self.db = session

            # 4. 初始化所有 Langchain 组件
            print("  🚀 Initializing Langchain components...")
            await langchain_manager.initialize()

            # 5. 获取 agent
            print("  🎯 Loading agent...")
            self.agent = get_gal_agent()

            # 6. 创建 chat service
            print("  💬 Creating chat service...")
            self.chat_service = ChatMessageService(self.db, self.agent)

            print("✅ System initialized successfully!")
            return True

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def upload_document(self, filepath: str) -> None:
        """上传文档到向量库"""
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            return

        print(f"Processing document: {filepath}")

        try:
            # 获取向量存储
            vectorstore = langchain_manager.get_vectorstore()

            # 读取文件
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # 简单分块（实际应该用 TextProcessor）
            chunks = []
            chunk_size = 1000
            for i in range(0, len(content), chunk_size):
                chunks.append(content[i:i + chunk_size])

            # 创建 Document 对象
            from langchain_core.documents import Document
            docs = [
                Document(
                    page_content=chunk,
                    metadata={
                        "filename": os.path.basename(filepath),
                        "chunk": idx
                    }
                )
                for idx, chunk in enumerate(chunks)
            ]

            # 添加到向量库
            ids = await vectorstore.aadd_documents(docs)

            print(f"Successfully processed: {os.path.basename(filepath)}")
            print(f"   Chunks: {len(ids)}")

        except Exception as e:
            print(f"Failed to process: {e}")

    async def ask_question(self, question: str) -> None:
        """提问（流式响应收集）- 支持连续对话"""
        print(f"\nQuestion: {question}")
        print("Thinking...")

        # 如果没有会话，创建一个新的
        if not self.current_session_code:
            self.current_session_code = str(uuid6.uuid7())
            await chat_session_crud.create(self.db, self.current_session_code)
            print(f"📝 创建新会话: {self.current_session_code}")
        else:
            print(f"💬 继续会话: {self.current_session_code}")

        full_answer = ""
        sources = []

        async for chunk in self.chat_service.chat(self.current_session_code, question):
            if chunk.startswith("data: "):
                try:
                    import json
                    data = json.loads(chunk[6:])
                    event = data.get("event")

                    if event == "message":
                        content = data["data"]["content"]
                        print(content, end="", flush=True)
                        full_answer += content
                    elif event == "reasoning":
                        tool_info = data["data"]
                        print(f"\n[使用工具: {tool_info.get('tool')}]", end="", flush=True)
                    elif event == "retrieval":
                        sources.append(data["data"])
                    elif event == "finish":
                        print()  # 换行
                except Exception as e:
                    # 忽略解析错误
                    pass

        print("\n")
        if sources:
            print(f"📚 引用来源 ({len(sources)}):")
            for i, source in enumerate(sources[:3], 1):
                filename = source.get("filename", "Unknown")
                similarity = source.get("similarity", 0)
                print(f"  {i}. {filename} (相似度: {similarity:.2f})")

    async def interactive_mode(self) -> None:
        print("\n" + "=" * 60)
        print("AI RAG System - Interactive Mode (LangGraph Agent)")
        print("=" * 60)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /upload   - Upload a document")
        print("  /new      - Start a new conversation session")
        print("  /exit     - Exit the program")
        print("\nJust type your question to ask.")
        print("=" * 60)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == '/exit':
                    print("Goodbye!")
                    break

                elif user_input.lower() == '/help':
                    print("Available commands:")
                    print("  /help     - Show help")
                    print("  /upload   - Upload document")
                    print("  /new      - Start a new conversation session")
                    print("  /exit     - Exit")
                    continue

                elif user_input.lower() == '/new':
                    self.current_session_code = None
                    print("✅ 已创建新会话，开始新的对话")
                    continue

                elif user_input.lower() == '/upload':
                    filepath = input("Enter file path: ").strip()
                    if filepath:
                        await self.upload_document(filepath)
                    continue

                await self.ask_question(user_input)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="AI RAG System CLI (LangGraph)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--upload", "-u", help="Upload a document")
    parser.add_argument("--question", "-q", help="Ask a single question")

    args = parser.parse_args()

    client = CLIClient()

    if not await client.initialize():
        print("Failed to initialize system. Please check:")
        print("  1. PostgreSQL RDS connection in .env")
        print("  2. API keys (DeepSeek/OpenAI)")
        print("  3. Network connectivity")
        return

    if args.upload:
        await client.upload_document(args.upload)

    elif args.question:
        await client.ask_question(args.question)

    elif args.interactive:
        await client.interactive_mode()

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())