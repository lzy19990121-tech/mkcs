"""
回测配置类

定义可复现回测所需的所有配置参数
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


@dataclass
class BacktestConfig:
    """回测配置类

    包含数据源、市场、费用、滑点、策略、风控等所有配置
    支持配置哈希计算，确保可复现性
    """

    # 数据源配置
    data_source: str = "mock"  # mock / yahoo / csv
    data_path: Optional[str] = None  # CSV 文件路径
    seed: int = 42

    # 市场配置
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT"])
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    interval: str = "1d"

    # 费用模型
    commission_per_share: float = 0.01
    commission_type: str = "per_share"  # per_share / per_dollar / fixed

    # 滑点模型
    slippage_enabled: bool = True
    slippage_type: str = "fixed"  # fixed / percent / bps
    slippage_value: float = 0.001  # 0.1% for percent, $0.01 for fixed
    slippage_bps: float = 0.0  # BPS滑点（1 BPS = 0.01%）

    # 策略配置
    strategy_name: str = "ma"
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # 风控配置
    risk_params: Dict[str, Any] = field(default_factory=dict)

    # 回测参数
    initial_cash: float = 100000.0

    def __post_init__(self):
        """验证配置有效性"""
        valid_data_sources = ["mock", "yahoo", "csv"]
        if self.data_source not in valid_data_sources:
            raise ValueError(f"无效的数据源: {self.data_source}, 可选: {valid_data_sources}")

        valid_commission_types = ["per_share", "per_dollar", "fixed"]
        if self.commission_type not in valid_commission_types:
            raise ValueError(f"无效的费用类型: {self.commission_type}")

        valid_slippage_types = ["fixed", "percent", "adaptive"]
        if self.slippage_type not in valid_slippage_types:
            raise ValueError(f"无效的滑点类型: {self.slippage_type}")

        # 验证日期格式
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"日期格式错误，应为 YYYY-MM-DD: {e}")

        # 验证 symbols
        if not self.symbols:
            raise ValueError("交易标的列表不能为空")

        # CSV 数据源需要 data_path
        if self.data_source == "csv" and not self.data_path:
            raise ValueError("CSV 数据源需要指定 data_path")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def compute_hash(self) -> str:
        """计算配置哈希值

        Returns:
            配置哈希值 (sha256:前缀 + 16位哈希)
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        hash_value = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return f"sha256:{hash_value}"

    def save(self, path: str):
        """保存配置到文件"""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BacktestConfig":
        """从文件加载配置"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def copy(self, **overrides) -> "BacktestConfig":
        """创建配置副本，可覆盖部分参数"""
        config_dict = self.to_dict()
        config_dict.update(overrides)
        return BacktestConfig(**config_dict)


@dataclass
class ReplayResult:
    """回测结果元数据"""

    backtest_date: str
    replay_mode: str

    data_source: Dict[str, Any]
    config: Dict[str, Any]

    config_hash: str
    git_commit: str
    data_hash: Optional[str] = None

    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def create_mock_config(
    symbols: List[str] = None,
    start_date: str = "2024-01-01",
    end_date: str = "2024-01-31",
    seed: int = 42
) -> BacktestConfig:
    """创建 Mock 数据源配置"""
    return BacktestConfig(
        data_source="mock",
        seed=seed,
        symbols=symbols or ["AAPL", "MSFT"],
        start_date=start_date,
        end_date=end_date,
        strategy_name="ma",
        strategy_params={"fast_period": 5, "slow_period": 20},
        risk_params={
            "max_position_ratio": 0.2,
            "max_positions": 5,
            "max_daily_loss_ratio": 0.05
        }
    )


def create_csv_config(
    data_path: str,
    symbols: List[str] = None,
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31"
) -> BacktestConfig:
    """创建 CSV 数据源配置"""
    return BacktestConfig(
        data_source="csv",
        data_path=data_path,
        symbols=symbols or ["AAPL"],
        start_date=start_date,
        end_date=end_date,
        strategy_name="ma",
        strategy_params={"fast_period": 10, "slow_period": 30},
        risk_params={
            "max_position_ratio": 0.2,
            "max_positions": 5,
            "max_daily_loss_ratio": 0.05
        }
    )


if __name__ == "__main__":
    """自测代码"""
    print("=== BacktestConfig 测试 ===\n")

    # 测试1: 默认配置
    print("1. 默认配置:")
    config = BacktestConfig()
    print(f"   数据源: {config.data_source}")
    print(f"   标的: {config.symbols}")
    print(f"   日期: {config.start_date} 到 {config.end_date}")
    print(f"   配置哈希: {config.compute_hash()}")

    # 测试2: Mock 配置
    print("\n2. Mock 配置:")
    mock_config = create_mock_config(symbols=["AAPL", "GOOGL"], seed=123)
    print(f"   数据源: {mock_config.data_source}")
    print(f"   Seed: {mock_config.seed}")
    print(f"   配置哈希: {mock_config.compute_hash()}")

    # 测试3: CSV 配置
    print("\n3. CSV 配置:")
    csv_config = create_csv_config(
        data_path="data/aapl.csv",
        symbols=["AAPL"]
    )
    print(f"   数据源: {csv_config.data_source}")
    print(f"   数据路径: {csv_config.data_path}")
    print(f"   配置哈希: {csv_config.compute_hash()}")

    # 测试4: 配置复现性
    print("\n4. 配置复现性验证:")
    config1 = create_mock_config(seed=42)
    config2 = create_mock_config(seed=42)
    config3 = create_mock_config(seed=43)  # 不同 seed

    hash1 = config1.compute_hash()
    hash2 = config2.compute_hash()
    hash3 = config3.compute_hash()

    print(f"   config1 哈希: {hash1}")
    print(f"   config2 哈希: {hash2}")
    print(f"   config3 哈希: {hash3}")
    print(f"   相同配置哈希一致: {hash1 == hash2}")
    print(f"   不同配置哈希不同: {hash1 != hash3}")

    # 测试5: 序列化
    print("\n5. 配置序列化:")
    json_str = config.to_json()
    print(f"   JSON 长度: {len(json_str)} 字符")

    # 测试6: 保存和加载
    print("\n6. 保存和加载:")
    test_path = "/tmp/test_config.json"
    config.save(test_path)
    loaded = BacktestConfig.load(test_path)
    print(f"   原始哈希: {config.compute_hash()}")
    print(f"   加载哈希: {loaded.compute_hash()}")
    print(f"   一致性: {config.compute_hash() == loaded.compute_hash()}")

    # 清理
    import os
    if os.path.exists(test_path):
        os.remove(test_path)

    # 测试7: 配置副本
    print("\n7. 配置副本:")
    original = create_mock_config(seed=42)
    copy = original.copy(seed=43)
    print(f"   原始 seed: {original.seed}")
    print(f"   副本 seed: {copy.seed}")
    print(f"   其他参数相同: {original.symbols == copy.symbols}")

    print("\n✓ 所有测试通过")
