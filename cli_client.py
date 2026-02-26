"""
AI RAG CLI client - 单表模式
所有文档存储到 ai_documents 表
"""

import asyncio
import sys
import os
import argparse
import logging
import codecs
import traceback
import json
import difflib
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Windows event loop setup
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Ensure UTF-8 stdout/stderr
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
stream_handler.encoding = 'utf-8'
root_logger.addHandler(stream_handler)
root_logger.setLevel(logging.INFO)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.chat_info import ChatMessageService
from app.core.langchain import langchain_manager
from app.core.db import async_db_manager, langchain_pool, db_initializer
from app.crud.chat_info import chat_session_crud
from app.services.ai.agent_graph import get_gal_agent
from app.services.retriever import RecursiveRetriever, RecursiveRetrieverConfig
from app.services.retriever.config import RecursiveRetrieverPresets
import uuid6
from sqlalchemy import text


class CLIClient:

    def __init__(self, workspace_root: Optional[str] = None):
        self.db = None
        self.agent = None
        self.chat_service = None
        self.current_session_code = None
        self.logger = logging.getLogger(__name__)
        self.workspace_root = Path(workspace_root or os.getcwd())
        self.memory_dir = self.workspace_root / 'session_memory'
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 递归检索配置
        self.recursive_retrieval_config = RecursiveRetrieverPresets.balanced()
        self.recursive_retriever = None
        self.enable_recursive_retrieval = True

    def safe_print(self, *args, **kwargs):
        try:
            print(*args, **kwargs)
        except UnicodeEncodeError:
            new_args = []
            for arg in args:
                if isinstance(arg, str):
                    new_args.append(arg.encode('ascii', errors='replace').decode('ascii'))
                else:
                    new_args.append(arg)
            print(*new_args, **kwargs)

    async def initialize(self) -> bool:
        self.safe_print("🚀 Initializing AI RAG system (单表模式)...")
        self.safe_print(f"🔄 Recursive Retrieval: {'Enabled' if self.enable_recursive_retrieval else 'Disabled'}")
        try:
            await async_db_manager.init_async_database()
            await langchain_pool.connect()
            await db_initializer.initialize()

            async with async_db_manager.get_async_db() as session:
                self.db = session

            await langchain_manager.initialize()

            # 尝试创建 agent
            try:
                self.agent = get_gal_agent()
                self.safe_print("✅ Agent created successfully")
            except Exception as agent_error:
                self.safe_print(f"⚠️  Warning: Agent creation failed: {agent_error}")
                self.agent = None

            if self.agent:
                self.chat_service = ChatMessageService(self.db, self.agent)

            # 初始化递归检索器
            self.recursive_retriever = RecursiveRetriever(
                config=self.recursive_retrieval_config,
                vectorstore=langchain_manager.get_vectorstore(),
            )

            self.safe_print("✅ System initialized successfully!")
            return True
        except Exception as e:
            self.safe_print(f"❌ Initialization failed: {e}")
            self.safe_print(traceback.format_exc())
            return False

    async def upload_document(self, filepath: str) -> None:
        """上传单个文件"""
        if not os.path.exists(filepath):
            self.safe_print(f"❌ Error: File not found: {filepath}")
            return

        self.safe_print(f"\n📄 Processing document: {filepath}")

        try:
            vectorstore = langchain_manager.get_vectorstore()

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = []
            chunk_size = 1000
            for i in range(0, len(content), chunk_size):
                chunks.append(content[i:i + chunk_size])

            from langchain_core.documents import Document
            docs = [
                Document(
                    page_content=chunk,
                    metadata={
                        "filename": os.path.basename(filepath),
                        "chunk": idx,
                        "uploaded_at": datetime.now().isoformat()
                    }
                )
                for idx, chunk in enumerate(chunks)
            ]

            ids = await vectorstore.aadd_documents(docs)

            self.safe_print(f"\n✅ Successfully processed: {os.path.basename(filepath)}")
            self.safe_print(f"   ✓ Chunks: {len(ids)}\n")

        except Exception as e:
            self.safe_print(f"❌ Failed to process: {e}")
            self.safe_print(traceback.format_exc())

    async def upload_directory(self, dirpath: str, extensions=None) -> None:
        """上传整个目录"""
        if extensions is None:
            extensions = ['.txt']
        
        dir_path = Path(dirpath)
        if not dir_path.exists() or not dir_path.is_dir():
            self.safe_print(f"❌ Error: Directory not found: {dirpath}")
            return
        
        # 获取所有匹配的文件
        files = []
        for ext in extensions:
            files.extend(dir_path.rglob(f"*{ext}"))
        
        if not files:
            self.safe_print(f"❌ No files found in: {dirpath}")
            return
        
        self.safe_print(f"📁 Found {len(files)} files in: {dirpath}")
        self.safe_print("-" * 50)
        
        success = 0
        failed = 0
        
        for filepath in files:
            try:
                await self.upload_document(str(filepath))
                success += 1
            except Exception as e:
                self.safe_print(f"❌ Failed: {filepath} - {e}")
                failed += 1
        
        self.safe_print("-" * 50)
        self.safe_print(f"✅ Upload complete: {success} success, {failed} failed")

    def _rerank_sources(self, question: str, sources: List[dict], top_n: int = 5) -> List[dict]:
        """基于余弦相似度对 sources 进行重排序"""
        if not sources:
            return []

        texts = []
        for src in sources:
            content = src.get('content') or src.get('page_content') or src.get('text') or src.get('filename', '')
            if isinstance(content, str):
                texts.append(content.strip())
            else:
                texts.append(str(content))

        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                lowercase=True,
                stop_words=None,
                analyzer='char',
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0
            )

            combined_texts = [question] + texts
            tfidf_matrix = vectorizer.fit_transform(combined_texts)

            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors)[0]

            scored = list(zip(similarities, sources))
            scored.sort(key=lambda x: x[0], reverse=True)

            return [src for _, src in scored[:top_n]]

        except Exception as e:
            self.logger.warning(f"余弦相似度计算失败: {e}")
            return self._rerank_sources_fallback(question, sources, top_n)

    def _rerank_sources_fallback(self, question: str, sources: List[dict], top_n: int = 5) -> List[dict]:
        """备选重排序方法"""
        scored = []
        for src in sources:
            content = src.get('content') or src.get('page_content') or src.get('text') or src.get('filename', '')
            ratio = difflib.SequenceMatcher(None, question, content).ratio()
            scored.append((ratio, src))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_n]]

    def _save_session_memory(self, session_code: str, entry: dict) -> None:
        path = self.memory_dir / f"session_{session_code}.json"
        data = []
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding='utf-8'))
            except Exception:
                data = []
        data.append(entry)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    def _load_session_memory(self, session_code: str) -> List[dict]:
        path = self.memory_dir / f"session_{session_code}.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return []

    async def ask_question(self, question: str) -> None:
        self.safe_print(f"\n{'='*60}")
        self.safe_print(f"Question: {question}")
        self.safe_print(f"{'='*60}")
        self.safe_print("🤔 Thinking...")

        if not self.current_session_code:
            self.current_session_code = str(uuid6.uuid7())
            await chat_session_crud.create(self.db, self.current_session_code)
            self.safe_print(f"📝 Created new session: {self.current_session_code}")
        else:
            self.safe_print(f"💬 Continue session: {self.current_session_code}")

        prior = self._load_session_memory(self.current_session_code)
        if prior:
            self.safe_print(f"🗃️ Loaded {len(prior)} memory entries for this session")

        full_answer = ""
        sources = []

        try:
            async for chunk in self.chat_service.chat(self.current_session_code, question):
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])
                        event = data.get("event")
                        if event == "message":
                            content = data["data"]["content"]
                            self.safe_print(content, end="", flush=True)
                            full_answer += content
                        elif event == "retrieval":
                            source = data["data"]
                            sources.append(source)
                        elif event == "finish":
                            self.safe_print()
                    except Exception as e:
                        self.safe_print(f"\n⚠️ Error processing chunk: {e}")
                        pass

            self.safe_print("\n")

            if sources:
                reranked = self._rerank_sources(question, sources, top_n=10)
                self.safe_print(f"📚 References (Top {len(reranked)}):")
                for i, source in enumerate(reranked, 1):
                    filename = source.get("filename", 'Unknown')
                    similarity = source.get("similarity") or 0
                    self.safe_print(f"  {i}. {filename} (similarity: {similarity})")

                mem_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'question': question,
                    'references': reranked
                }
                self._save_session_memory(self.current_session_code, mem_entry)

        except Exception as e:
            self.safe_print(f"\n❌ Error during chat: {e}")
            self.safe_print(traceback.format_exc())

    async def interactive_mode(self) -> None:
        self.safe_print("\n" + "=" * 60)
        self.safe_print("AI RAG System - Interactive Mode (单表模式)")
        self.safe_print("=" * 60)
        self.safe_print("Commands:")
        self.safe_print("  /help      - Show this help")
        self.safe_print("  /upload    - Upload a document")
        self.safe_print("  /uploaddir - Upload all files in a directory")
        self.safe_print("  /new       - Start a new conversation session")
        self.safe_print("  /retrieve  - Toggle recursive retrieval (on/off)")
        self.safe_print("  /depth     - Set recursion max depth (1-4)")
        self.safe_print("  /preset    - Choose retrieval preset (light/balanced/deep)")
        self.safe_print("  /exit      - Exit the program")
        self.safe_print("\nJust type your question to ask.")
        self.safe_print("=" * 60)

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue
                if user_input.lower() == '/exit':
                    self.safe_print("Goodbye!")
                    break
                elif user_input.lower() == '/help':
                    self.safe_print("Available commands: /help /upload /uploaddir /new /retrieve /depth /preset /exit")
                    continue
                elif user_input.lower() == '/new':
                    self.current_session_code = None
                    self.safe_print("✅ 已创建新会话，开始新的对话")
                    continue
                elif user_input.lower() == '/retrieve':
                    self.enable_recursive_retrieval = not self.enable_recursive_retrieval
                    status = "Enabled ✅" if self.enable_recursive_retrieval else "Disabled ❌"
                    self.safe_print(f"🔄 Recursive Retrieval: {status}")
                    continue
                elif user_input.lower() == '/preset':
                    self.safe_print("Choose retrieval preset:")
                    self.safe_print("  1. light   - Fast, shallow retrieval (depth=2)")
                    self.safe_print("  2. balanced - Recommended default (depth=3)")
                    self.safe_print("  3. deep    - Deep exploration (depth=4)")
                    choice = input("Preset#: ").strip()
                    if choice == '1':
                        self.recursive_retrieval_config = RecursiveRetrieverPresets.light()
                        self.safe_print("✅ Switched to LIGHT preset")
                    elif choice == '2':
                        self.recursive_retrieval_config = RecursiveRetrieverPresets.balanced()
                        self.safe_print("✅ Switched to BALANCED preset")
                    elif choice == '3':
                        self.recursive_retrieval_config = RecursiveRetrieverPresets.deep()
                        self.safe_print("✅ Switched to DEEP preset")
                    else:
                        self.safe_print("Invalid choice")
                    if self.recursive_retriever:
                        self.recursive_retriever.config = self.recursive_retrieval_config
                    continue
                elif user_input.lower() == '/depth':
                    depth_str = input("Set max recursion depth (1-4): ").strip()
                    if depth_str.isdigit() and 1 <= int(depth_str) <= 4:
                        self.recursive_retrieval_config.max_recursion_depth = int(depth_str)
                        if self.recursive_retriever:
                            self.recursive_retriever.config = self.recursive_retrieval_config
                        self.safe_print(f"✅ Max recursion depth set to: {depth_str}")
                    else:
                        self.safe_print("Invalid depth (must be 1-4)")
                    continue
                elif user_input.lower() == '/upload':
                    filepath = input("Enter file path: ").strip()
                    if filepath:
                        await self.upload_document(filepath)
                    continue
                elif user_input.lower() == '/uploaddir':
                    dirpath = input("Enter directory path: ").strip()
                    if dirpath:
                        await self.upload_directory(dirpath)
                    continue
                else:
                    await self.ask_question(user_input)

            except KeyboardInterrupt:
                self.safe_print("\n\nExiting...")
                break
            except Exception as e:
                self.safe_print(f"Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="AI RAG System CLI (单表模式)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--upload", "-u", help="Upload a document")
    parser.add_argument("--uploaddir", "-d", help="Upload all files in a directory")
    parser.add_argument("--question", "-q", help="Ask a single question")
    parser.add_argument("--extensions", "-e", nargs="+", default=[".txt"], help="File extensions to upload")

    args = parser.parse_args()

    client = CLIClient()

    if not await client.initialize():
        client.safe_print("Failed to initialize system. Please check your configuration and .env settings.")
        return

    if args.uploaddir:
        await client.upload_directory(args.uploaddir, args.extensions)
    elif args.upload:
        await client.upload_document(args.upload)
    elif args.question:
        await client.ask_question(args.question)
    elif args.interactive:
        await client.interactive_mode()
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
