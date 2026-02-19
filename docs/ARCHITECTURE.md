# 项目架构

## 文件结构

```
gal_helper_backend/
├── src/gal_helper_backend/          # 主源代码包
│   ├── __init__.py
│   ├── main.py                      # FastAPI 应用入口
│   ├── cli.py                       # CLI 客户端（包含 rerank 逻辑）
│   ├── api/                         # FastAPI 路由
│   ├── core/                        # 核心功能模块
│   │   ├── db.py                    # 数据库连接
│   │   ├── langchain.py             # Langchain 集成
│   │   └── config.py                # 配置管理
│   ├── crud/                        # 数据库 CRUD 操作
│   ├── models/                      # SQLAlchemy ORM 模型
│   ├── schemas/                     # Pydantic 数据验证
│   ├── services/                    # 业务逻辑层
│   ├── reranker/                    # Rerank 模块
│   │   ├── __init__.py
│   │   ├── tfidf_reranker.py        # TF-IDF 实现
│   │   └── base.py                  # 基类定义
│   └── utils/                       # 工具函数
│
├── tests/                           # 单元测试
│   ├── conftest.py                  # Pytest 配置
│   ├── test_rerank.py               # Rerank 测试
│   ├── test_connection.py
│   └── test_vector.py
│
├── scripts/                         # 工作脚本
│   ├── migrate_documents.py         # 文档迁移脚本
│   └── setup_service.py             # 服务部署脚本
│
├── tools/                           # 开发工具
│   ├── verify_implementation.py     # 实施验证
│   └── compare_rerank.py            # Rerank 方案对比
│
├── docs/                            # 文档
│   ├── RERANK_REFERENCE.md          # Rerank 完整参考
│   └── ARCHITECTURE.md              # 本文件
│
├── README.md                        # 项目主说明
├── pyproject.toml                   # 项目配置
├── .env                             # 环境变量（git ignore）
├── .gitignore
└── uv.lock                          # 依赖锁定文件
```

## 核心模块说明

### CLI 模块（`src/gal_helper_backend/cli.py`）

**功能**：RAG 系统的命令行客户端

**关键类**：
- `DialogTopic`：对话主题枚举（资源查找、运行问题、相关工具、游戏资讯）
- `CLIClient`：主客户端类

**关键方法**：
- `_rerank_sources()`：基于余弦相似度的文档排序
- `_rerank_sources_fallback()`：降级排序（使用 difflib）
- `ask_question()`：处理用户问题
- `upload_document()`：上传文档

### Reranker 模块（`src/gal_helper_backend/reranker/`）

**功能**：文档排序和相关性计算

**实现**：
- TF-IDF 向量化
- 余弦相似度计算
- 自动降级机制

### 依赖关系

```
CLIClient
  ├── ChatMessageService
  ├── AsyncDatabaseManager
  │   └── SQLAlchemy AsyncSession
  ├── LangchainManager
  │   └── PostgresCheckpointStorage
  ├── ChatSessionCRUD
  └── TF-IDF Reranker
      ├── scikit-learn.TfidfVectorizer
      └── scikit-learn.cosine_similarity
```

## 数据流

### 1. 提问流程

```
用户输入问题
    ↓
CLIClient.ask_question()
    ↓
Langchain Agent 查询相关文档
    ↓
Rerank 排序相关文档
    ├─ TF-IDF 向量化
    ├─ 计算余弦相似度
    └─ 返回 top-n 结果
    ↓
生成回答
    ↓
保存到数据库
    ↓
返回给用户
```

### 2. 文档上传流程

```
上传文件
    ↓
解析文档
    ↓
分块处理
    ↓
向量化（Embedding）
    ↓
存储到向量数据库
    ↓
关联到特定主题表
    ↓
确认完成
```

## 技术栈

### 核心框架
- **Web Framework**：FastAPI
- **ORM**：SQLAlchemy 2.0 (async)
- **数据库**：PostgreSQL + pgvector

### AI/ML
- **LLM Framework**：Langchain + Langgraph
- **向量模型**：DeepSeek Embedding API
- **Chat 模型**：DeepSeek Chat API
- **Rerank**：scikit-learn（TF-IDF + 余弦相似度）

### 工具和库
- **包管理**：uv（超快的 Python 包管理器）
- **异步**：asyncio + asyncpg
- **数据验证**：Pydantic
- **类型检查**：Python 3.10+

## 配置管理

### 环境变量（`.env`）

```env
# LLM 配置
CHAT_MODEL_BASE_URL=https://api.deepseek.com
CHAT_MODEL_NAME=deepseek-chat
CHAT_MODEL_API_KEY=...

# Embedding 配置
BASE_EMBEDDING_MODEL_BASE_URL=https://api.deepseek.com
BASE_EMBEDDING_MODEL_NAME=deepseek-chat
BASE_EMBEDDING_API_KEY=...

# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ai_knowledge_db
DB_USER=postgres
DB_PASSWORD=...

# 异步数据库 URL
ASYNC_DATABASE_URL=postgresql+asyncpg://...
LANGCHAIN_DATABASE_URL=postgresql://...
```

### 项目配置（`pyproject.toml`）

**核心依赖**：
- fastapi, uvicorn：Web 框架
- sqlalchemy[asyncio], asyncpg：数据库
- langchain, langgraph：AI 框架
- scikit-learn：Rerank 向量计算

**可选依赖**：
- sentence-transformers：更高级的重排序（future）
- torch：深度学习支持（future）

## 扩展点

### 1. Reranker 扩展

当前实现：TF-IDF + 余弦相似度

可扩展方向：
1. BM25 重排序
2. Sentence Transformer 向量化
3. Learning to Rank（LTR）模型
4. 多度量融合

### 2. 主题扩展

当前主题：
- 资源查找
- 运行问题
- 相关工具与软件
- 游戏资讯

扩展方式：
1. 在 `DialogTopic` 枚举中添加新主题
2. 在数据库中创建对应的向量存储表
3. 在 `DialogTopic.get_table_name()` 中映射新主题

### 3. 对话能力扩展

当前使用：Langchain + DeepSeek

可扩展方向：
1. 集成其他 LLM（GPT, Claude 等）
2. 添加工具调用能力
3. 实现多轮对话记忆
4. 添加反馈学习机制

## 部署考虑

### 开发环境
```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate      # Windows

# 安装依赖
uv pip install -e ".[all]"

# 运行应用
python -m uvicorn src.gal_helper_backend.main:app --reload
```

### 生产环境
参考 `scripts/setup_service.py` 进行 systemd 服务配置

### Docker 部署（未来）
- 创建 Dockerfile
- 编写 docker-compose.yml
- 配置环境变量管理

## 监控和日志

**日志配置**：
- 使用 Python 标准库 logging
- 配置日志级别和输出格式
- 记录异常堆栈跟踪

**监控指标**（推荐）：
- 文档检索相关性（用户点击率）
- 响应时间（P50, P95, P99）
- 错误率和降级率
- Rerank 排序准确性

## 安全考虑

- 敏感信息存储在 `.env`（git ignore）
- 数据库连接使用异步和连接池
- API 认证和授权（TODO）
- 输入验证（Pydantic schemas）

---

**版本**：1.0  
**最后更新**：2026-02-19
