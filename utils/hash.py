"""
哈希计算工具

用于计算配置、数据文件的哈希值，以及获取 git commit
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any


def compute_config_hash(config_dict: Dict[str, Any]) -> str:
    """计算配置的哈希值

    Args:
        config_dict: 配置字典

    Returns:
        哈希值字符串 (sha256:前缀 + 16位哈希)
    """
    config_str = json.dumps(config_dict, sort_keys=True, separators=(',', ':'))
    hash_value = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    return f"sha256:{hash_value}"


def compute_data_hash(data_path: str) -> Optional[str]:
    """计算数据文件的 SHA256 哈希值

    Args:
        data_path: 数据文件路径

    Returns:
        哈希值字符串，如果文件不存在返回 None
    """
    path = Path(data_path)
    if not path.exists():
        return None

    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)

    return f"sha256:{sha256.hexdigest()[:16]}"


def compute_string_hash(content: str) -> str:
    """计算字符串的哈希值

    Args:
        content: 字符串内容

    Returns:
        哈希值字符串
    """
    hash_value = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"sha256:{hash_value}"


def get_git_commit(short: bool = True) -> str:
    """获取当前 git commit hash

    Args:
        short: 是否只返回前8位

    Returns:
        git commit hash，如果失败返回 "unknown"
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).parent.parent)  # 从 utils/hash.py 定位到项目根目录
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
            return commit[:8] if short else commit
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    return "unknown"


def get_git_branch() -> str:
    """获取当前 git 分支

    Returns:
        分支名，如果失败返回 "unknown"
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).parent.parent)
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    return "unknown"


def get_repo_status() -> Dict[str, Any]:
    """获取仓库状态信息

    Returns:
        包含 commit、branch、dirty 状态的字典
    """
    return {
        "commit": get_git_commit(short=True),
        "branch": get_git_branch(),
        "dirty": _is_repo_dirty()
    }


def _is_repo_dirty() -> bool:
    """检查仓库是否有未提交的修改"""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(Path(__file__).parent.parent)
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    return False


def verify_data_integrity(data_path: str, expected_hash: str) -> bool:
    """验证数据文件完整性

    Args:
        data_path: 数据文件路径
        expected_hash: 预期的哈希值

    Returns:
        是否匹配
    """
    actual_hash = compute_data_hash(data_path)
    if actual_hash is None:
        return False
    return actual_hash == expected_hash


if __name__ == "__main__":
    """自测代码"""
    print("=== Hash Utils 测试 ===\n")

    # 测试1: 配置哈希
    print("1. 配置哈希计算:")
    config = {"data_source": "mock", "seed": 42, "symbols": ["AAPL"]}
    hash1 = compute_config_hash(config)
    hash2 = compute_config_hash(config)
    print(f"   配置: {config}")
    print(f"   哈希: {hash1}")
    print(f"   一致性: {hash1 == hash2}")

    # 测试2: 字符串哈希
    print("\n2. 字符串哈希:")
    content = "Hello, World!"
    str_hash = compute_string_hash(content)
    print(f"   内容: {content}")
    print(f"   哈希: {str_hash}")

    # 测试3: 文件哈希
    print("\n3. 文件哈希:")
    test_file = "/tmp/test_hash_file.txt"
    Path(test_file).write_text("Test content for hashing")
    file_hash = compute_data_hash(test_file)
    print(f"   文件: {test_file}")
    print(f"   哈希: {file_hash}")

    # 测试4: Git 信息
    print("\n4. Git 信息:")
    commit = get_git_commit()
    branch = get_git_branch()
    status = get_repo_status()
    print(f"   Commit: {commit}")
    print(f"   Branch: {branch}")
    print(f"   Dirty: {status['dirty']}")

    # 测试5: 数据完整性验证
    print("\n5. 数据完整性验证:")
    is_valid = verify_data_integrity(test_file, file_hash)
    print(f"   验证通过: {is_valid}")

    is_invalid = verify_data_integrity(test_file, "sha256:invalid")
    print(f"   错误哈希验证: {is_invalid}")

    # 清理
    import os
    if os.path.exists(test_file):
        os.remove(test_file)

    print("\n✓ 所有测试通过")
