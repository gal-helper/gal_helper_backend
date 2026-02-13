import asyncio
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.chat_info import ChatMessageService
from app.core.langchain import langchain_manager
from app.core.db import async_db_manager, langchain_pool
from app.crud.chat_info import chat_session_crud
from app.services.ai.agent_graph import get_gal_agent
import uuid6


class CLIClient:

    def __init__(self):
        self.db = None
        self.agent = None
        self.chat_service = None

    async def initialize(self) -> bool:
        print("Initializing AI RAG system...")
        try:
            await async_db_manager.init_async_database()
            await langchain_pool.connect()

            await langchain_manager.init_langchain_manager()

            async_db_context = async_db_manager.get_async_db()
            self.db = await anext(async_db_context.__aiter__())

            self.agent = await get_gal_agent()

            self.chat_service = ChatMessageService(self.db, self.agent)

            print("System initialized successfully")
            return True
        except Exception as e:
            print(f"Initialization failed: {e}")
            return False

    async def upload_document(self, filepath: str) -> None:
        """上传文档到向量库"""
        if not os.path.exists(filepath):
            print(f"Error: File not found: {filepath}")
            return

        print(f"Processing document: {filepath}")

        try:
            # 获取向量存储
            vectorstore = await langchain_manager.get_vectorstore()

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
        """提问（流式响应收集）"""
        print(f"\nQuestion: {question}")
        print("Thinking...")

        # 创建临时会话
        session_code = str(uuid6.uuid7())
        await chat_session_crud.create(self.db, session_code)

        full_answer = ""
        sources = []

        async for chunk in self.chat_service.chat(session_code, question):
            if chunk.startswith("data: "):
                try:
                    import json
                    data = json.loads(chunk[6:])
                    if data.get("event") == "message":
                        content = data["data"]["content"]
                        print(content, end="", flush=True)
                        full_answer += content
                    elif data.get("event") == "retrieval":
                        sources.append(data["data"])
                except:
                    pass

        print("\n")
        if sources:
            print(f"Sources ({len(sources)}):")
            for i, source in enumerate(sources[:3], 1):
                filename = source.get("filename", "Unknown")
                print(f"  {i}. {filename}")

    async def interactive_mode(self) -> None:
        print("\n" + "=" * 60)
        print("AI RAG System - Interactive Mode (LangGraph Agent)")
        print("=" * 60)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /upload   - Upload a document")
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
                    print("  /exit     - Exit")
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