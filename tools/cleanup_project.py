"""
项目文件清理和重组脚本

此脚本整理项目结构为企业级标准：
- 删除冗余文件
- 整理文档到 docs/
- 整理测试到 tests/
- 整理工具到 tools/ 和 scripts/
"""

import os
import shutil
from pathlib import Path


def get_files_to_delete():
    """返回应该删除的文件列表"""
    return [
        # 示例和测试文件
        "111.py",                           # 示例文件
        "00_START_HERE.txt",               # 临时说明文件
        "REFACTORING_PLAN.md",             # 重构计划（已完成）
        
        # 冗余总结文档
        "COMPLETION_REPORT.txt",           # 完成报告（重复）
        "FINAL_SUMMARY.txt",               # 最终总结（重复）
        "IMPLEMENTATION_SUMMARY.md",       # 实施总结（重复）
        
        # Rerank 文档（已移到 docs/）
        "RERANK_IMPROVEMENT.md",
        "RERANK_QUICK_REFERENCE.md",
        "README_RERANK.md",
        
        # 根目录的测试文件（已移到 tests/）
        "test_rerank.py",
        "compare_rerank_methods.py",       # 已移到 tools/
        "verify_rerank_implementation.py", # 已移到 tools/
        
        # CLI 冗余文件
        "refactor_cli_client",             # 整个目录（cli_client.py 已在根目录）
    ]


def get_files_to_move():
    """返回应该移动的文件映射"""
    return {
        # 目前这些文件已经在正确的位置
        # 或将在脚本逻辑中处理
    }


def safe_delete(file_path, dry_run=True):
    """安全删除文件或文件夹"""
    path = Path(file_path)
    
    if not path.exists():
        print(f"  ℹ️  {file_path} - 不存在，跳过")
        return
    
    if dry_run:
        if path.is_dir():
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            print(f"  [删除] {file_path}/ - 目录（约 {size/1024:.1f} KB）")
        else:
            size = path.stat().st_size
            print(f"  [删除] {file_path} - ({size/1024:.1f} KB)")
    else:
        if path.is_dir():
            shutil.rmtree(path)
            print(f"  ✓ 删除目录：{file_path}")
        else:
            path.unlink()
            print(f"  ✓ 删除文件：{file_path}")


def show_cleanup_plan(project_root=None):
    """显示清理计划（模拟）"""
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    print("\n" + "="*70)
    print("文件清理计划（模拟运行 - 不实际删除）")
    print("="*70)
    
    files_to_delete = get_files_to_delete()
    
    print("\n将删除的文件/目录：\n")
    total_size = 0
    for filename in files_to_delete:
        file_path = project_root / filename
        safe_delete(file_path, dry_run=True)
    
    print("\n" + "="*70)
    print("清理计划预览完成")
    print("="*70)
    print("\n执行清理：")
    print("  python tools/cleanup_project.py --execute")
    print("\n❌ 警告：此操作不可逆！请确保已备份重要文件")


def execute_cleanup(project_root=None):
    """执行实际的清理操作"""
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    print("\n" + "="*70)
    print("执行文件清理")
    print("="*70)
    
    files_to_delete = get_files_to_delete()
    
    print("\n删除文件...\n")
    for filename in files_to_delete:
        file_path = project_root / filename
        safe_delete(file_path, dry_run=False)
    
    print("\n" + "="*70)
    print("✓ 清理完成")
    print("="*70)


def show_new_structure():
    """显示新的项目结构"""
    print("\n" + "="*70)
    print("新项目结构")
    print("="*70)
    
    structure = """
gal_helper_backend/
├── 📄 README.md                    ← 项目主说明
├── 📄 pyproject.toml               ← 项目配置
├── 📄 .env                         ← 环境变量
│
├── 📂 src/
│   └── gal_helper_backend/         ← 主源代码包
│       ├── main.py
│       ├── cli.py (重命名自 cli_client.py)
│       ├── api/, core/, crud/, models/
│       ├── schemas/, services/, reranker/, utils/
│       └── __init__.py
│
├── 📂 tests/                       ← 单元测试
│   ├── test_rerank.py              (✓ 已更新)
│   ├── test_connection.py
│   ├── test_vector.py
│   └── conftest.py
│
├── 📂 scripts/                     ← 工作脚本
│   ├── migrate_documents.py
│   └── setup_service.py
│
├── 📂 tools/                       ← 开发工具
│   ├── verify_implementation.py     (✓ 已创建)
│   └── compare_rerank.py
│
├── 📂 docs/                        ← 文档
│   ├── RERANK_REFERENCE.md         (✓ 已创建 - 完整参考)
│   └── ARCHITECTURE.md             (✓ 已创建 - 架构设计)
│
└── 📂 [deleted]/
    ├── 111.py                      ✗ 删除
    ├── 00_START_HERE.txt           ✗ 删除
    ├── COMPLETION_REPORT.txt       ✗ 删除
    ├── FINAL_SUMMARY.txt           ✗ 删除
    ├── IMPLEMENTATION_SUMMARY.md   ✗ 删除
    ├── RERANK_*.md                 ✗ 删除 (已合并到 docs/)
    ├── refactor_cli_client/        ✗ 删除 (内容重复)
    └── 根目录的各种脚本             ✗ 移到合适位置
"""
    
    print(structure)
    print("="*70)


if __name__ == "__main__":
    import sys
    
    project_root = Path(__file__).parent.parent
    
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        # 实际执行清理
        print("\n⚠️  确认执行清理操作？这将删除大量文件。")
        print("输入 'YES' 确认：", end=" ")
        confirm = input().strip().upper()
        
        if confirm == "YES":
            execute_cleanup(project_root)
            show_new_structure()
        else:
            print("操作已取消")
    else:
        # 显示清理计划
        show_cleanup_plan(project_root)
        show_new_structure()
