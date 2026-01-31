"""
运行清单管理

实现实验资产化：每次运行输出到 runs/<experiment_id>/ 目录
experiment_id 通过配置的稳定哈希生成
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


def generate_experiment_id(config: Dict[str, Any], git_commit: str) -> str:
    """生成稳定的实验ID

    Args:
        config: 配置字典
        git_commit: Git commit hash

    Returns:
        实验ID (8位哈希)
    """
    # 创建稳定的配置字符串（排除时间戳等动态字段）
    stable_fields = {
        'data_source': config.get('data_source'),
        'data_path': config.get('data_path'),
        'seed': config.get('seed'),
        'symbols': sorted(config.get('symbols', [])),
        'start_date': config.get('start_date'),
        'end_date': config.get('end_date'),
        'interval': config.get('interval'),
        'commission_per_share': config.get('commission_per_share'),
        'slippage_bps': config.get('slippage_bps'),
        'strategy_name': config.get('strategy_name'),
        'strategy_params': config.get('strategy_params'),
        'git_commit': git_commit
    }

    # 生成哈希
    config_str = json.dumps(stable_fields, sort_keys=True)
    hash_value = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    return f"exp_{hash_value}"


@dataclass
class RunManifest:
    """运行清单"""

    # 实验标识
    experiment_id: str
    started_at: str = ""
    ended_at: Optional[str] = None
    status: str = "running"  # running / completed / failed

    # Git信息
    git_commit: str = ""
    git_branch: str = "main"
    repo_dirty: bool = False

    # 配置哈希
    config_hash: str = ""
    data_hash: Optional[str] = None

    # 运行配置
    mode: str = "replay_mock"  # replay_mock / replay_real / paper
    symbols: List[str] = field(default_factory=list)
    start_date: str = ""
    end_date: str = ""
    interval: str = "1d"

    # 成本模型
    cost_model: Dict[str, Any] = field(default_factory=dict)

    # 产物列表
    artifacts: List[str] = field(default_factory=list)

    # 性能指标
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def save(self, run_dir: Path):
        """保存到运行目录"""
        manifest_path = run_dir / "run_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, run_dir: Path) -> "RunManifest":
        """从运行目录加载"""
        manifest_path = run_dir / "run_manifest.json"
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(**data)

    def mark_completed(self, metrics: Dict[str, Any], artifacts: List[str]):
        """标记运行完成"""
        self.status = "completed"
        self.ended_at = datetime.now().isoformat()
        self.metrics = metrics
        self.artifacts = artifacts

    def mark_failed(self, error: str):
        """标记运行失败"""
        self.status = "failed"
        self.ended_at = datetime.now().isoformat()
        self.metrics = {"error": error}


def create_run_directory(
    base_dir: str = "runs",
    config: Optional[Dict[str, Any]] = None,
    git_commit: Optional[str] = None
) -> tuple[Path, RunManifest]:
    """创建运行目录

    Args:
        base_dir: 基础目录
        config: 配置字典
        git_commit: Git commit hash

    Returns:
        (run_dir, manifest)
    """
    if git_commit is None:
        from utils.hash import get_git_commit
        git_commit = get_git_commit(short=True)

    if config is None:
        config = {}

    # 生成实验ID
    experiment_id = generate_experiment_id(config, git_commit)

    # 创建运行目录
    run_dir = Path(base_dir) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 创建清单
    manifest = RunManifest(
        experiment_id=experiment_id,
        git_commit=git_commit,
        config_hash=generate_experiment_id(config, git_commit).replace("exp_", "cfg_"),
        mode=config.get("mode", "replay_mock"),
        symbols=config.get("symbols", []),
        start_date=config.get("start_date", ""),
        end_date=config.get("end_date", ""),
        interval=config.get("interval", "1d"),
        cost_model={
            "commission_per_share": config.get("commission_per_share", 0.01),
            "slippage_bps": config.get("slippage_bps", 0.0)
        }
    )
    manifest.started_at = datetime.now().isoformat()

    # 保存清单
    manifest.save(run_dir)

    return run_dir, manifest


def list_runs(base_dir: str = "runs", status: Optional[str] = None) -> List[RunManifest]:
    """列出所有运行

    Args:
        base_dir: 基础目录
        status: 过滤状态（可选）

    Returns:
        运行清单列表
    """
    runs = []
    base_path = Path(base_dir)

    if not base_path.exists():
        return runs

    for exp_dir in base_path.iterdir():
        if not exp_dir.is_dir():
            continue

        manifest_path = exp_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue

        try:
            manifest = RunManifest.load(exp_dir)
            if status is None or manifest.status == status:
                runs.append(manifest)
        except Exception:
            continue

    # 按开始时间排序
    runs.sort(key=lambda m: m.started_at, reverse=True)
    return runs


if __name__ == "__main__":
    """测试代码"""
    print("=== RunManifest 测试 ===\n")

    # 测试生成实验ID
    config = {
        "data_source": "mock",
        "seed": 42,
        "symbols": ["AAPL", "MSFT"],
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "strategy_name": "ma",
        "strategy_params": {"fast": 5, "slow": 20}
    }

    exp_id = generate_experiment_id(config, "abc123")
    print(f"实验ID: {exp_id}")

    # 测试创建运行目录
    run_dir, manifest = create_run_directory(
        base_dir="runs/test",
        config=config,
        git_commit="abc123"
    )

    print(f"运行目录: {run_dir}")
    print(f"清单文件: {run_dir / 'run_manifest.json'}")

    # 测试标记完成
    manifest.mark_completed(
        metrics={"total_return": 0.05, "trade_count": 10},
        artifacts=["summary.json", "trades.csv", "equity_curve.csv"]
    )
    manifest.save(run_dir)

    # 测试加载
    loaded = RunManifest.load(run_dir)
    print(f"\n加载状态: {loaded.status}")
    print(f"产物: {loaded.artifacts}")

    # 清理
    import shutil
    if Path("runs/test").exists():
        shutil.rmtree("runs/test")

    print("\n✓ 测试通过")
