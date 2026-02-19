# 递归检索快速参考卡

## 🚀 30 秒快速开始

### CLI 方式
```bash
python -m cli_client --interactive

# 输入命令
/retrieve              # 启用递归检索
/preset balanced       # 选择预设
你的问题              # 自动进行递归检索
```

### API 方式
```bash
curl -X POST http://localhost:8000/api/v1/search/recursive-search \
  -d "query=你的问题&preset=balanced"
```

### Python 方式
```python
from app.services.retriever import RecursiveRetriever
from app.services.retriever.config import RecursiveRetrieverPresets

config = RecursiveRetrieverPresets.balanced()
retriever = RecursiveRetriever(config=config, vectorstore=vs)
results, report = await retriever.retrieve("你的问题")
```

---

## 📚 4 个预设一览

| 预设 | 用途 | 深度 | 速度 | 精度 |
|------|------|------|------|------|
| **Light** | ⚡ 快速查询 | 2 | 🚀 1s | ⭐⭐ |
| **Balanced** | ⭐ 推荐 | 3 | 🚴 3s | ⭐⭐⭐⭐ |
| **Deep** | 🔬 深入研究 | 4 | 🐢 5s | ⭐⭐⭐⭐⭐ |
| **Single** | 📌 关闭递归 | 1 | 🚀 0.8s | ⭐⭐ |

---

## 🎮 CLI 命令速查

```
/retrieve           切换递归检索 开/关
/preset light       选择预设 (light/balanced/deep)
/depth 3            设置最大深度 (1-4)
/topic              选择检索主题
/upload <file>      上传文档
/new                新建会话
/help               显示帮助
/exit               退出
```

---

## 🔧 常用参数配置

### 速度优化
```python
config = RecursiveRetrieverPresets.light()
# 或
config.max_recursion_depth = 2
config.initial_k = 5
```

### 准确度优化
```python
config = RecursiveRetrieverPresets.deep()
# 或
config.max_recursion_depth = 4
config.initial_k = 15
config.rerank_method = "cross_encoder"
```

### 自定义组合
```python
config = RecursiveRetrieverConfig(
    max_recursion_depth=3,        # 深度
    initial_k=10,                 # 初始文档数
    min_confidence_score=0.6,      # 递归触发阈值
    rerank_method="cosine",        # 重排序方法
)
```

---

## 📊 API 端点速查

### 预设检索
```
POST /api/v1/search/recursive-search
参数: query, preset, topic, enable_logging
```

### 自定义检索
```
POST /api/v1/search/recursive-search/custom
参数: query, max_depth, initial_k, final_k, rerank_method, ...
```

### 获取预设列表
```
GET /api/v1/search/recursive-search/presets
返回: 所有可用预设的详细信息
```

---

## 💡 故障排查速查

| 问题 | 解决方案 |
|------|---------|
| 🐢 太慢 | 用 Light 预设或 `/depth 2` |
| ❌ 相关性差 | 用 Deep 预设或 `rerank_method=cross_encoder` |
| 🔄 没有递归 | `/retrieve` 启用或降低 `min_confidence_score` |
| 💾 内存高 | 降低 `initial_k` 或 `max_recursion_depth` |
| ❓ 子问题差 | 禁用自动生成或提高置信度阈值 |

---

## 📈 结果解读

```json
{
    "retrieval_depth": 3,           // 实际使用的深度
    "total_results": 28,            // 收集的总结果数
    "final_results": 5,             // 最终返回数
    "execution_time": 2.45,         // 执行耗时 (秒)
    "relevance_score": 0.92,        // 相关性分数 (0-1)
    "retrieval_path": [             // 检索路径
        "原始查询: 如何修复?",
        "子问题: DirectX 错误"
    ]
}
```

---

## 🎯 选择建议

```
┌─ 问题复杂度?
│  ├─ 简单 (事实查询) ──> Single Layer
│  ├─ 中等 (一般问题) ──> Balanced ⭐
│  └─ 复杂 (多层次) ──> Deep
│
├─ 对速度的要求?
│  ├─ < 1s ──> Single Layer / Light
│  ├─ < 5s ──> Light / Balanced
│  └─ > 5s ──> Deep 可接受
│
└─ 对准确度的要求?
   ├─ 基础 (70%+) ──> Light
   ├─ 优秀 (85%+) ──> Balanced ⭐
   └─ 最佳 (90%+) ──> Deep
```

---

## 📚 详细资源

| 资源 | 位置 | 内容 |
|------|------|------|
| 完整指南 | `docs/RECURSIVE_RETRIEVAL.md` | 3000+ 字详细说明 |
| API 文档 | `app/api/v1/recursive_search.py` | REST 接口详情 |
| 单元测试 | `tests/test_recursive_retrieval.py` | 14 个测试用例 |
| 演示脚本 | `tools/demo_recursive_retrieval.py` | 8 个场景演示 |

---

## ✅ 检查清单

- [ ] 已启用递归检索 (`/retrieve`)
- [ ] 已选择合适的预设或自定义配置
- [ ] 已测试不同的深度和参数
- [ ] 已检查执行时间是否可接受
- [ ] 已验证结果的相关性
- [ ] 已阅读完整文档了解更多选项

---

## 🔗 快速链接

- 📖 [完整用户指南](../docs/RECURSIVE_RETRIEVAL.md)
- 🔗 [REST API 文档](../app/api/v1/recursive_search.py)
- 🧪 [单元测试](../tests/test_recursive_retrieval.py)
- 🎯 [演示脚本](../tools/demo_recursive_retrieval.py)
- 📋 [实现总结](../RECURSIVE_RETRIEVAL_IMPLEMENTATION.md)

---

**版本**: 1.1.0 | **最后更新**: 2026-02-19 | **状态**: ✅ 生产就绪
