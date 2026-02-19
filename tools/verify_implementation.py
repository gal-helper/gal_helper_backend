"""
验证脚本：检查项目实施是否完整

检查清单：
- 代码导入和修改
- 依赖配置
- 文档完整性
"""

import sys
from pathlib import Path


def check_imports():
    """检查必要的导入"""
    print("✓ 检查导入...")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        print("  ✓ scikit-learn 正确导入")
        print("  ✓ numpy 正确导入")
        return True
    except ImportError as e:
        print(f"  ✗ 导入失败：{e}")
        return False


def check_source_code():
    """检查源代码中的 rerank 实现"""
    print("\n✓ 检查源代码...")
    
    # 检查 cli.py（或 cli_client.py）
    project_root = Path(__file__).parent.parent
    cli_files = list(project_root.glob("**/cli.py")) + list(project_root.glob("**/cli_client.py"))
    
    if not cli_files:
        print("  ✗ 未找到 CLI 文件")
        return False
    
    cli_path = cli_files[0]
    content = cli_path.read_text(encoding='utf-8')
    
    checks = {
        "TfidfVectorizer 导入": "TfidfVectorizer" in content,
        "cosine_similarity 导入": "cosine_similarity" in content,
        "_rerank_sources 方法": "def _rerank_sources(" in content,
        "容错机制": "_rerank_sources_fallback(" in content,
    }
    
    all_passed = True
    for check_name, result in checks.items():
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {check_name}")
        all_passed = all_passed and result
    
    return all_passed


def check_dependencies():
    """检查依赖配置"""
    print("\n✓ 检查依赖配置...")
    
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        print("  ✗ pyproject.toml 不存在")
        return False
    
    content = pyproject_path.read_text(encoding='utf-8')
    
    if "scikit-learn" in content:
        print("  ✓ scikit-learn 已添加到依赖")
        return True
    else:
        print("  ✗ scikit-learn 未在依赖中")
        return False


def check_documentation():
    """检查文档完整性"""
    print("\n✓ 检查文档...")
    
    project_root = Path(__file__).parent.parent
    docs = {
        "docs/RERANK_REFERENCE.md": "Rerank 完整参考",
        "docs/ARCHITECTURE.md": "架构文档",
        "README.md": "项目 README",
    }
    
    all_exist = True
    for doc_path, description in docs.items():
        full_path = project_root / doc_path
        exists = full_path.exists()
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {doc_path} ({description})")
        all_exist = all_exist and exists
    
    return all_exist


def main():
    print("="*70)
    print("项目实施验证")
    print("="*70)
    
    results = {
        "导入检查": check_imports(),
        "源代码检查": check_source_code(),
        "依赖配置": check_dependencies(),
        "文档检查": check_documentation(),
    }
    
    print("\n" + "="*70)
    print("验证结果")
    print("="*70)
    
    for check_name, result in results.items():
        symbol = "✓" if result else "✗"
        status = "通过" if result else "失败"
        print(f"{symbol} {check_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ 所有验证通过！项目已准备好使用")
        print("\n后续步骤：")
        print("1. pip install scikit-learn>=1.3.0")
        print("2. pytest tests/")
        print("3. 启动应用进行集成测试")
    else:
        print("✗ 部分验证失败，请检查上面的详细信息")
    print("="*70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
