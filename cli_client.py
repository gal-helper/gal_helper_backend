"""
Refactored AI RAG CLI client with Recursive Retrieval
- 精简重复定义
- 四个预筛选接口（资源查找/运行问题/相关工具与软件/游戏资讯）
- 基于余弦相似度（TF-IDF）的高级重排序（rerank）功能，支持中文
- 递归检索（Recursive Retrieval）：多层级文档检索，自动生成子问题
- 简单的会话记忆持久化到本地文件夹（session_memory）
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
from enum import Enum
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

# Local project imports (kept the same names as original project)
from app.services.chat_info import ChatMessageService
from app.core.langchain import langchain_manager
from app.core.db import async_db_manager, langchain_pool, db_initializer
from app.crud.chat_info import chat_session_crud
from app.services.ai.agent_graph import get_gal_agent
from app.services.retriever import RecursiveRetriever, RecursiveRetrieverConfig
from app.services.retriever.config import RecursiveRetrieverPresets
import uuid6
from sqlalchemy import text


# ==================== 主题枚举（四个接口） ====================
class DialogTopic(Enum):
    RESOURCE = "资源查找"
    TECHNICAL = "运行问题"
    TOOLS = "相关工具与软件"
    NEWS = "游戏资讯"

    @classmethod
    def get_table_name(cls, topic: 'DialogTopic') -> str:
        mapping = {
            cls.RESOURCE: "vectorstore_resource",
            cls.TECHNICAL: "vectorstore_technical",
            cls.TOOLS: "vectorstore_tools",
            cls.NEWS: "vectorstore_news",
        }
        return mapping[topic]

    @classmethod
    def from_string(cls, value: str) -> Optional['DialogTopic']:
        for topic in cls:
            if topic.value == value or topic.name.lower() == str(value).lower():
                return topic
        return None


class CLIClient:

    def __init__(self, topic: DialogTopic = DialogTopic.RESOURCE, workspace_root: Optional[str] = None):
        self.db = None
        self.agent = None
        self.chat_service = None
        self.current_session_code = None
        self.current_topic = topic
        self.logger = logging.getLogger(__name__)
        self.workspace_root = Path(workspace_root or os.getcwd())
        self.memory_dir = self.workspace_root / 'refactor_cli_client' / 'session_memory'
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
        self.safe_print("🚀 Initializing AI RAG system with recursive retrieval support...")
        self.safe_print(f"📌 Current Topic: {self.current_topic.value}")
        self.safe_print(f"🔄 Recursive Retrieval: {'Enabled' if self.enable_recursive_retrieval else 'Disabled'}")
        try:
            await async_db_manager.init_async_database()
            await langchain_pool.connect()
            await db_initializer.initialize()
            await self._create_topic_tables()

            async with async_db_manager.get_async_db() as session:
                self.db = session

            await langchain_manager.initialize()
            
            # 尝试创建 agent，但如果失败则继续（使用递归检索代替）
            try:
                self.agent = get_gal_agent()
                self.safe_print("✅ Agent created successfully")
            except Exception as agent_error:
                self.safe_print(f"⚠️  Warning: Agent creation failed, using retrieval mode only: {agent_error}")
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

    async def _create_topic_tables(self) -> None:
        async with async_db_manager.get_async_db() as session:
            try:
                for topic in DialogTopic:
                    table_name = DialogTopic.get_table_name(topic)
                    result = await session.execute(
                        text(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}')")
                    )
                    if not result.scalar():
                        await session.execute(
                            text(f"""
                                CREATE TABLE {table_name} (
                                    id SERIAL PRIMARY KEY,
                                    content TEXT NOT NULL,
                                    embedding vector(1536),
                                    filename VARCHAR(255),
                                    topic VARCHAR(50),
                                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                    metadata JSONB
                                )
                            """ )
                        )
                        await session.execute(
                            text(f"CREATE INDEX idx_{table_name}_embedding ON {table_name} USING ivfflat (embedding vector_cosine_ops)")
                        )
                        self.logger.info(f"Created table: {table_name}")
                await session.commit()
            except Exception as e:
                self.logger.warning(f"Table creation issue (may already exist): {e}")
                await session.rollback()

    async def upload_document(self, filepath: str, target_topic: Optional[DialogTopic] = None) -> None:
        if not os.path.exists(filepath):
            self.safe_print(f"❌ Error: File not found: {filepath}")
            return

        topic = target_topic or self.current_topic
        table_name = DialogTopic.get_table_name(topic)

        self.safe_print(f"\n📄 Processing document: {filepath}")
        self.safe_print(f"   📌 Topic: {topic.value}")
        self.safe_print(f"   📊 Table: {table_name}")

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
                        "topic": topic.value,
                        "table": table_name,
                        "uploaded_at": datetime.now().isoformat()
                    }
                )
                for idx, chunk in enumerate(chunks)
            ]

            ids = await vectorstore.aadd_documents(docs)

            self.safe_print(f"\n✅ Successfully processed: {os.path.basename(filepath)}")
            self.safe_print(f"   ✓ Chunks: {len(ids)}")
            self.safe_print(f"   ✓ Topic: {topic.value}")
            self.safe_print(f"   ✓ Stored in: {table_name}\n")

        except Exception as e:
            self.safe_print(f"❌ Failed to process: {e}")
            self.safe_print(traceback.format_exc())

    def _rerank_sources(self, question: str, sources: List[dict], top_n: int = 5) -> List[dict]:
        """
        基于余弦相似度对 sources 进行重排序，返回 top_n。
        
        使用 TF-IDF 向量化文本，计算查询与每个文档的余弦相似度。
        余弦相似度范围为 [0, 1]，值越大表示相似度越高。
        
        Args:
            question: 查询问题
            sources: 候选源列表
            top_n: 返回的最多结果数
            
        Returns:
            按余弦相似度排序的 sources 列表
        """
        if not sources:
            return []
        
        # 提取文本内容
        texts = []
        for src in sources:
            content = src.get('content') or src.get('page_content') or src.get('text') or src.get('filename', '')
            # 清理文本
            if isinstance(content, str):
                texts.append(content.strip())
            else:
                texts.append(str(content))
        
        # 构建 TF-IDF 向量化器
        # 最多考虑 100 个特征（词汇），预先分词避免特殊字符问题
        try:
            vectorizer = TfidfVectorizer(
                max_features=100,
                lowercase=True,
                stop_words=None,  # 不移除停用词，保留所有词汇
                analyzer='char',  # 使用字符级别的分析，支持中文
                ngram_range=(1, 2),  # 单字和双字
                min_df=1,  # 至少在1个文档中出现
                max_df=1.0  # 最多在100%的文档中出现
            )
            
            # 组合查询和文档进行向量化
            combined_texts = [question] + texts
            tfidf_matrix = vectorizer.fit_transform(combined_texts)
            
            # 计算查询与每个文档的余弦相似度
            query_vector = tfidf_matrix[0:1]  # 第一行是查询
            doc_vectors = tfidf_matrix[1:]    # 其余行是文档
            
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            # 创建 (相似度, 源) 对并排序
            scored = list(zip(similarities, sources))
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # 返回 top_n
            return [src for _, src in scored[:top_n]]
            
        except Exception as e:
            self.logger.warning(f"余弦相似度计算失败，降级使用 difflib: {e}")
            # 如果 TF-IDF 计算失败，降级为 difflib 实现
            return self._rerank_sources_fallback(question, sources, top_n)
    
    async def _recursive_retrieve(self, question: str, topic: Optional[str] = None) -> tuple:
        """
        执行递归检索（新功能）
        
        Returns:
            (检索结果列表, 检索报告)
        """
        if not self.enable_recursive_retrieval or not self.recursive_retriever:
            return [], None
        
        try:
            self.safe_print("\n🔄 Performing recursive retrieval...")
            results, report = await self.recursive_retriever.retrieve(
                question,
                topic=topic or DialogTopic.get_table_name(self.current_topic),
                return_report=True,
            )
            
            if report:
                self.safe_print(f"   ✓ Recursion Depth: {report.recursion_depth_used}/{self.recursive_retrieval_config.max_recursion_depth}")
                self.safe_print(f"   ✓ Total Results Collected: {report.total_results}")
                self.safe_print(f"   ✓ Final Results After Dedup: {report.final_results}")
                self.safe_print(f"   ✓ Execution Time: {report.execution_time:.2f}s")
            
            return results, report
        except Exception as e:
            self.logger.warning(f"递归检索失败: {e}")
            return [], None
    
    def _rerank_sources_fallback(self, question: str, sources: List[dict], top_n: int = 5) -> List[dict]:
        """
        备选的重排序方法，使用 difflib 的 SequenceMatcher。
        当 TF-IDF 向量化失败时使用此方法。
        """
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
        self.safe_print(f"Topic: {self.current_topic.value}")
        self.safe_print(f"{'='*60}")
        self.safe_print("🤔 Thinking...")

        if not self.current_session_code:
            self.current_session_code = str(uuid6.uuid7())
            await chat_session_crud.create(self.db, self.current_session_code)
            self.safe_print(f"📝 Created new session: {self.current_session_code}")
        else:
            self.safe_print(f"💬 Continue session: {self.current_session_code}")

        # Load prior memory and display brief context
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
                            if source.get("topic") == self.current_topic.value or source.get('table') == DialogTopic.get_table_name(self.current_topic):
                                sources.append(source)
                        elif event == "finish":
                            self.safe_print()
                    except Exception as e:
                        self.safe_print(f"\n⚠️ Error processing chunk: {e}")
                        pass

            self.safe_print("\n")

            if sources:
                # 重排序 sources
                reranked = self._rerank_sources(question, sources, top_n=10)
                self.safe_print(f"📚 References (Top {len(reranked)} after rerank):")
                for i, source in enumerate(reranked, 1):
                    filename = source.get("filename", source.get('meta', {}).get('filename', 'Unknown'))
                    similarity = source.get("similarity") or 0
                    topic = source.get("topic") or source.get('table') or 'Unknown'
                    self.safe_print(f"  {i}. {filename} (Reported sim: {similarity}) [Topic: {topic}]")

                # 持久化本次检索结果到会话记忆
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
        self.safe_print("AI RAG System - Interactive Mode with Recursive Retrieval")
        self.safe_print("=" * 60)
        self.safe_print("Commands:")
        self.safe_print("  /help      - Show this help")
        self.safe_print("  /upload    - Upload a document")
        self.safe_print("  /new       - Start a new conversation session")
        self.safe_print("  /topic     - Select a topic (预筛选接口)")
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
                    self.safe_print("Available commands: /help /upload /new /topic /retrieve /depth /preset /exit")
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
                    # 更新检索器配置
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
                        # 允许在上传时选择主题
                        self.safe_print("Choose topic (number) or press Enter to use current:")
                        for i, t in enumerate(DialogTopic, 1):
                            self.safe_print(f"  {i}. {t.value}")
                        choice = input("Topic#: ").strip()
                        target = None
                        if choice.isdigit() and 1 <= int(choice) <= len(DialogTopic):
                            target = list(DialogTopic)[int(choice)-1]
                        await self.upload_document(filepath, target)
                    continue
                elif user_input.lower() == '/topic':
                    self.safe_print("Select topic:")
                    for i, t in enumerate(DialogTopic, 1):
                        self.safe_print(f"  {i}. {t.value}")
                    choice = input("Topic#: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(DialogTopic):
                        self.current_topic = list(DialogTopic)[int(choice)-1]
                        self.safe_print(f"✅ Current topic set to: {self.current_topic.value}")
                    else:
                        self.safe_print("Invalid choice")
                    continue
                else:
                    await self.ask_question(user_input)

            except KeyboardInterrupt:
                self.safe_print("\n\nExiting...")
                break
            except Exception as e:
                self.safe_print(f"Error: {e}")


async def main():
    parser = argparse.ArgumentParser(description="AI RAG System CLI (Refactored)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--upload", "-u", help="Upload a document")
    parser.add_argument("--question", "-q", help="Ask a single question")
    parser.add_argument("--topic", "-t", help="Initial topic (name or value)")

    args = parser.parse_args()

    init_topic = DialogTopic.RESOURCE
    if args.topic:
        t = DialogTopic.from_string(args.topic)
        if t:
            init_topic = t

    client = CLIClient(topic=init_topic)

    if not await client.initialize():
        client.safe_print("Failed to initialize system. Please check your configuration and .env settings.")
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