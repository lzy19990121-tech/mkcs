"""
SPL-6a-A: 漂移对象定义

定义需要漂移检测的对象及其统计口径。
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pathlib import Path
import yaml
import numpy as np
import pandas as pd


class MetricType(Enum):
    """指标类型"""
    DISTRIBUTION = "distribution"  # 分布型指标（需要分桶）
    STATISTICAL = "statistical"    # 统计型指标（数值比较）


class DriftCategory(Enum):
    """漂移类别"""
    INPUT_DISTRIBUTION = "input_distribution"   # 输入分布漂移
    RISK_BEHAVIOR = "risk_behavior"             # 风险行为漂移
    MODEL_BEHAVIOR = "model_behavior"           # 模型/规则漂移
    PORTFOLIO_BEHAVIOR = "portfolio_behavior"   # 组合层面漂移


@dataclass
class Bucket:
    """分桶定义"""
    name: str
    range: tuple  # (lower, upper)

    def contains(self, value: float) -> bool:
        """检查值是否在桶内"""
        lower, upper = self.range
        # 处理无穷大
        if lower == -np.inf:
            return value < upper
        if upper == np.inf:
            return value >= lower
        return lower <= value < upper


@dataclass
class DriftObjectConfig:
    """单个漂移对象配置"""
    name: str
    category: DriftCategory
    metric_type: MetricType
    description: str
    window_days: List[int]  # [detection_window, baseline_window]
    buckets: Optional[List[Bucket]] = None
    percentiles: Optional[List[float]] = None
    confidence_levels: Optional[List[float]] = None
    metrics: List[str] = field(default_factory=list)

    def __post_init__(self):
        """验证配置"""
        if self.metric_type == MetricType.DISTRIBUTION and not self.buckets:
            raise ValueError(f"Distribution metric {self.name} requires buckets")

    def get_baseline_window(self) -> int:
        """获取基线窗口天数"""
        return self.window_days[-1]

    def get_detection_window(self) -> int:
        """获取检测窗口天数"""
        return self.window_days[0]


@dataclass
class DriftSnapshot:
    """漂移对象的统计快照"""
    object_name: str
    category: DriftCategory
    timestamp: datetime
    window_days: int

    # 分布型数据
    bucket_counts: Dict[str, int] = field(default_factory=dict)
    bucket_proportions: Dict[str, float] = field(default_factory=dict)

    # 统计型数据
    statistics: Dict[str, float] = field(default_factory=dict)
    percentiles: Dict[float, float] = field(default_factory=dict)

    # 样本信息
    sample_size: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "object_name": self.object_name,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "window_days": self.window_days,
            "bucket_counts": self.bucket_counts,
            "bucket_proportions": self.bucket_proportions,
            "statistics": self.statistics,
            "percentiles": self.percentiles,
            "sample_size": self.sample_size,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None
        }


@dataclass
class DriftResult:
    """漂移检测结果"""
    object_name: str
    category: DriftCategory
    metric_name: str  # PSI, KS test, etc.
    drift_value: float  # 漂移程度
    p_value: Optional[float] = None  # 统计显著性
    status: str = "GREEN"  # GREEN/YELLOW/RED
    threshold: Optional[float] = None  # 使用的阈值
    message: str = ""

    # 详细信息
    baseline_snapshot: Optional[DriftSnapshot] = None
    current_snapshot: Optional[DriftSnapshot] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "object_name": self.object_name,
            "category": self.category.value,
            "metric_name": self.metric_name,
            "drift_value": self.drift_value,
            "p_value": self.p_value,
            "status": self.status,
            "threshold": self.threshold,
            "message": self.message,
            "baseline_snapshot": self.baseline_snapshot.to_dict() if self.baseline_snapshot else None,
            "current_snapshot": self.current_snapshot.to_dict() if self.current_snapshot else None,
            "details": self.details
        }


class DriftObjectRegistry:
    """漂移对象注册表"""

    def __init__(self, config_path: Optional[Path] = None):
        """初始化注册表

        Args:
            config_path: 配置文件路径（默认使用 drift_objects.yaml）
        """
        self.config_path = config_path or Path("config/drift_objects.yaml")
        self.objects: Dict[str, DriftObjectConfig] = {}
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Drift objects config not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # 解析配置
        for category_name, category_objects in config.get("drift_objects", {}).items():
            category = DriftCategory(category_name)

            for obj_name, obj_config in category_objects.items():
                # 解析 buckets
                buckets = None
                if "buckets" in obj_config:
                    buckets = [
                        Bucket(
                            name=b["name"],
                            range=(b["range"][0], b["range"][1])
                        )
                        for b in obj_config["buckets"]
                    ]

                # 创建配置对象
                drift_obj = DriftObjectConfig(
                    name=obj_name,
                    category=category,
                    metric_type=MetricType(obj_config.get("metric_type", "statistical")),
                    description=obj_config.get("description", ""),
                    window_days=obj_config.get("window_days", [30, 90]),
                    buckets=buckets,
                    percentiles=obj_config.get("percentiles"),
                    confidence_levels=obj_config.get("confidence_levels"),
                    metrics=obj_config.get("metrics", [])
                )

                self.objects[f"{category_name}.{obj_name}"] = drift_obj

    def get_object(self, name: str) -> DriftObjectConfig:
        """获取漂移对象配置"""
        if name not in self.objects:
            raise ValueError(f"Drift object not found: {name}")
        return self.objects[name]

    def list_objects(self, category: Optional[DriftCategory] = None) -> List[str]:
        """列出漂移对象"""
        if category:
            return [
                name for name, obj in self.objects.items()
                if obj.category == category
            ]
        return list(self.objects.keys())

    def get_objects_by_category(self, category: DriftCategory) -> List[DriftObjectConfig]:
        """按类别获取漂移对象"""
        return [
            obj for obj in self.objects.values()
            if obj.category == category
        ]


if __name__ == "__main__":
    """测试漂移对象定义"""
    print("=== SPL-6a-A: 漂移对象定义测试 ===\n")

    # 加载注册表
    registry = DriftObjectRegistry()

    # 列出所有对象
    print("所有漂移对象:")
    for name in registry.list_objects():
        obj = registry.get_object(name)
        print(f"  - {name}: {obj.description}")

    # 按类别统计
    print(f"\n按类别统计:")
    for category in DriftCategory:
        objects = registry.get_objects_by_category(category)
        print(f"  {category.value}: {len(objects)} 个对象")

    # 测试单个对象
    print(f"\n示例对象: input_distribution.returns")
    returns_obj = registry.get_object("input_distribution.returns")
    print(f"  类型: {returns_obj.metric_type.value}")
    print(f"  窗口: {returns_obj.window_days}")
    print(f"  桶: {[b.name for b in returns_obj.buckets]}")
    print(f"  指标: {returns_obj.metrics}")

    print("\n✅ 漂移对象定义测试通过")
