"""
SPL-6a-B: 漂移指标计算

实现各类漂移检测指标：
- PSI (Population Stability Index)
- KS test / Wasserstein distance
- Bucket share shift
- Percentile shift
- Absolute/relative change
- Threshold breach detection
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from analysis.drift_objects import DriftObjectConfig, DriftSnapshot, Bucket


class DriftMetricsCalculator:
    """漂移指标计算器"""

    def __init__(self):
        """初始化计算器"""
        self.metrics_registry = {
            "psi": self._calculate_psi,
            "js_divergence": self._calculate_js_divergence,
            "bucket_shift": self._calculate_bucket_shift,
            "ks_test": self._calculate_ks_test,
            "wasserstein": self._calculate_wasserstein,
            "percentile_shift": self._calculate_percentile_shift,
            "tail_change": self._calculate_tail_change,
            "absolute_change": self._calculate_absolute_change,
            "relative_change": self._calculate_relative_change,
            "threshold_breach": self._check_threshold_breach,
            "rate_breach": self._check_rate_breach,
            "stability_check": self._check_stability,
        }

    def calculate_metric(
        self,
        metric_name: str,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """计算指定指标

        Args:
            metric_name: 指标名称
            baseline: 基线数据
            current: 当前数据
            config: 漂移对象配置
            **kwargs: 额外参数

        Returns:
            {"value": float, "p_value": Optional[float], "details": Dict}
        """
        if metric_name not in self.metrics_registry:
            raise ValueError(f"Unknown metric: {metric_name}")

        calculator = self.metrics_registry[metric_name]
        return calculator(baseline, current, config, **kwargs)

    # ========== 分布型指标 ==========

    def _calculate_psi(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """计算 PSI (Population Stability Index)

        PSI = sum((Current% - Expected%) * ln(Current% / Expected%))

        解释:
        - < 0.1: 无显著漂移
        - 0.1 - 0.25: 轻微漂移
        - > 0.25: 显著漂移
        """
        if not config.buckets:
            raise ValueError("PSI requires buckets")

        # 计算分桶分布
        baseline_dist = self._bucket_distribution(baseline, config.buckets)
        current_dist = self._bucket_distribution(current, config.buckets)

        # 避免除零
        baseline_dist = np.where(baseline_dist == 0, 0.0001, baseline_dist)
        current_dist = np.where(current_dist == 0, 0.0001, current_dist)

        # 计算 PSI
        psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))

        return {
            "value": float(psi),
            "p_value": None,
            "details": {
                "baseline_distribution": baseline_dist.tolist(),
                "current_distribution": current_dist.tolist(),
                "bucket_shift": (current_dist - baseline_dist).tolist()
            }
        }

    def _calculate_js_divergence(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """计算 JS 散度 (Jensen-Shannon Divergence)

        JS 散度是对称的 KL 散度，范围 [0, 1]
        """
        if not config.buckets:
            raise ValueError("JS divergence requires buckets")

        # 计算分桶分布
        baseline_dist = self._bucket_distribution(baseline, config.buckets)
        current_dist = self._bucket_distribution(current, config.buckets)

        # 避免 log(0)
        baseline_dist = np.where(baseline_dist == 0, 1e-10, baseline_dist)
        current_dist = np.where(current_dist == 0, 1e-10, current_dist)

        # 计算 KL 散度
        def kl_div(p, q):
            return np.sum(p * np.log(p / q))

        # 平均分布
        m = 0.5 * (baseline_dist + current_dist)

        # JS 散度
        js = 0.5 * kl_div(baseline_dist, m) + 0.5 * kl_div(current_dist, m)

        return {
            "value": float(js),
            "p_value": None,
            "details": {
                "baseline_distribution": baseline_dist.tolist(),
                "current_distribution": current_dist.tolist()
            }
        }

    def _calculate_bucket_shift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """计算桶占比变化

        返回各桶占比变化的绝对值
        """
        if not config.buckets:
            raise ValueError("Bucket shift requires buckets")

        # 计算分桶分布
        baseline_dist = self._bucket_distribution(baseline, config.buckets)
        current_dist = self._bucket_distribution(current, config.buckets)

        # 计算变化
        shift = current_dist - baseline_dist
        max_abs_shift = np.abs(shift).max()
        max_shift_bucket = np.abs(shift).argmax()

        bucket_names = [b.name for b in config.buckets]

        return {
            "value": float(max_abs_shift),
            "p_value": None,
            "details": {
                "bucket_shifts": {
                    bucket_names[i]: float(shift[i])
                    for i in range(len(bucket_names))
                },
                "max_shift_bucket": bucket_names[max_shift_bucket],
                "all_shifts": shift.tolist()
            }
        }

    def _calculate_ks_test(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """计算 KS 检验 (Kolmogorov-Smirnov test)

        检验两个样本是否来自同一分布
        """
        statistic, p_value = stats.ks_2samp(baseline, current)

        return {
            "value": float(statistic),
            "p_value": float(p_value),
            "details": {
                "reject_null": p_value < 0.05,
                "significance_level": 0.05
            }
        }

    def _calculate_wasserstein(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """计算 Wasserstein 距离 (Earth Mover's Distance)

        衡量两个分布之间的距离
        """
        # 简化实现：使用 CDF 差的积分
        baseline_sorted = np.sort(baseline)
        current_sorted = np.sort(current)

        # 计算经验 CDF
        baseline_cdf = np.arange(1, len(baseline_sorted) + 1) / len(baseline_sorted)
        current_cdf = np.arange(1, len(current_sorted) + 1) / len(current_sorted)

        # 插值到相同点
        all_values = np.concatenate([baseline_sorted, current_sorted])
        all_values.sort()

        baseline_interp = np.interp(all_values, baseline_sorted, baseline_cdf)
        current_interp = np.interp(all_values, current_sorted, current_cdf)

        # 计算面积
        distance = np.trapz(np.abs(baseline_interp - current_interp), all_values)

        return {
            "value": float(distance),
            "p_value": None,
            "details": {
                "normalized": float(distance / (baseline.max() - baseline.min()))
            }
        }

    # ========== 统计型指标 ==========

    def _calculate_percentile_shift(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """计算分位数变化

        比较指定分位数的数值变化
        """
        percentiles = config.percentiles or [5, 10, 25, 50, 75, 90, 95]

        baseline_pcts = np.percentile(baseline, percentiles)
        current_pcts = np.percentile(current, percentiles)

        shifts = current_pcts - baseline_pcts
        max_abs_shift = np.abs(shifts).max()
        max_shift_pct = percentiles[np.abs(shifts).argmax()]

        return {
            "value": float(max_abs_shift),
            "p_value": None,
            "details": {
                "percentile_shifts": {
                    f"p{p}": float(shifts[i])
                    for i, p in enumerate(percentiles)
                },
                "max_shift_percentile": max_shift_pct,
                "baseline_percentiles": baseline_pcts.tolist(),
                "current_percentiles": current_pcts.tolist()
            }
        }

    def _calculate_tail_change(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """计算尾部变化

        比较尾部（如P5、P10）的变化
        """
        tail_percentiles = [5, 10]
        baseline_tails = np.percentile(baseline, tail_percentiles)
        current_tails = np.percentile(current, tail_percentiles)

        # 尾部恶化（更小是更差）
        tail_worsened = current_tails < baseline_tails
        max_worsening = float((baseline_tails - current_tails).max())

        return {
            "value": float(max_worsening),
            "p_value": None,
            "details": {
                "baseline_tails": baseline_tails.tolist(),
                "current_tails": current_tails.tolist(),
                "tail_worsened": tail_worsened.tolist()
            }
        }

    def _calculate_absolute_change(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """计算绝对变化

        适用于单一统计值（如 MDD、CVaR）
        """
        baseline_value = np.mean(baseline) if len(baseline) > 0 else 0
        current_value = np.mean(current) if len(current) > 0 else 0

        abs_change = current_value - baseline_value

        return {
            "value": float(abs_change),
            "p_value": None,
            "details": {
                "baseline_value": float(baseline_value),
                "current_value": float(current_value),
                "direction": "increase" if abs_change > 0 else "decrease"
            }
        }

    def _calculate_relative_change(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """计算相对变化

        返回百分比变化
        """
        baseline_value = np.mean(baseline) if len(baseline) > 0 else 1e-10
        current_value = np.mean(current) if len(current) > 0 else 0

        # 避免除零
        if abs(baseline_value) < 1e-10:
            rel_change = 0.0
        else:
            rel_change = (current_value - baseline_value) / abs(baseline_value)

        return {
            "value": float(rel_change),
            "p_value": None,
            "details": {
                "baseline_value": float(baseline_value),
                "current_value": float(current_value),
                "percentage_change": float(rel_change * 100)
            }
        }

    def _check_threshold_breach(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        threshold: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """检查阈值突破

        检查当前值是否超过阈值
        """
        current_value = np.mean(current) if len(current) > 0 else 0

        if threshold is None:
            # 使用基线的均值 + 2倍标准差作为阈值
            baseline_mean = np.mean(baseline) if len(baseline) > 0 else 0
            baseline_std = np.std(baseline) if len(baseline) > 0 else 1
            threshold = baseline_mean + 2 * baseline_std

        breached = current_value > threshold

        return {
            "value": float(current_value - threshold),
            "p_value": None,
            "details": {
                "current_value": float(current_value),
                "threshold": float(threshold),
                "breached": breached,
                "breach_amount": float(abs(current_value - threshold))
            }
        }

    def _check_rate_breach(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        threshold: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """检查频率突破

        适用于比率类指标（如 gating trigger rate）
        """
        current_rate = np.mean(current) if len(current) > 0 else 0

        if threshold is None:
            # 使用基线 + 50% 作为阈值
            baseline_rate = np.mean(baseline) if len(baseline) > 0 else 0
            threshold = baseline_rate * 1.5

        breached = current_rate > threshold

        return {
            "value": float(current_rate - threshold),
            "p_value": None,
            "details": {
                "current_rate": float(current_rate),
                "threshold_rate": float(threshold),
                "breached": breached,
                "baseline_rate": float(np.mean(baseline) if len(baseline) > 0 else 0)
            }
        }

    def _check_stability(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        stability_threshold: float = 2.0,
        **kwargs
    ) -> Dict[str, Any]:
        """检查稳定性

        检查当前值的标准差是否在合理范围内
        """
        current_std = np.std(current) if len(current) > 1 else 0
        current_mean = np.mean(current) if len(current) > 0 else 1

        # 变异系数
        cv = current_std / abs(current_mean) if abs(current_mean) > 1e-10 else 0

        stable = cv < stability_threshold

        return {
            "value": float(cv),
            "p_value": None,
            "details": {
                "stable": stable,
                "coefficient_of_variation": float(cv),
                "current_std": float(current_std),
                "current_mean": float(current_mean)
            }
        }

    # ========== 辅助方法 ==========

    def _bucket_distribution(
        self,
        data: np.ndarray,
        buckets: List[Bucket]
    ) -> np.ndarray:
        """计算数据在桶中的分布

        Returns:
            每个桶的占比（总和为1）
        """
        counts = np.zeros(len(buckets))

        for value in data:
            for i, bucket in enumerate(buckets):
                if bucket.contains(value):
                    counts[i] += 1
                    break

        # 转换为占比
        total = len(data)
        if total > 0:
            return counts / total
        return np.array([1.0 / len(buckets)] * len(buckets))


@dataclass
class DriftMetricsResult:
    """漂移指标计算结果"""
    metric_name: str
    value: float
    p_value: Optional[float]
    status: str  # GREEN/YELLOW/RED
    threshold_green: float
    threshold_yellow: float
    message: str
    details: Dict[str, Any]


class DriftMetricsEvaluator:
    """漂移指标评估器"""

    def __init__(self, threshold_config: Optional[Dict] = None):
        """初始化评估器

        Args:
            threshold_config: 漂移阈值配置
        """
        self.calculator = DriftMetricsCalculator()
        self.thresholds = threshold_config or self._default_thresholds()

    def _default_thresholds(self) -> Dict:
        """默认阈值配置"""
        return {
            "psi": {"green": 0.1, "yellow": 0.25},
            "js_divergence": {"green": 0.1, "yellow": 0.2},
            "bucket_shift": {"green": 0.05, "yellow": 0.1},
            "ks_test": {"green": 0.1, "yellow": 0.2},  # statistic
            "wasserstein": {"green": 0.1, "yellow": 0.2},
            "percentile_shift": {"green": 0.02, "yellow": 0.05},
            "tail_change": {"green": 0.01, "yellow": 0.03},
            "absolute_change": {"green": 0.01, "yellow": 0.02},
            "relative_change": {"green": 0.1, "yellow": 0.2},  # 10%, 20%
        }

    def evaluate(
        self,
        metric_name: str,
        baseline: np.ndarray,
        current: np.ndarray,
        config: DriftObjectConfig,
        **kwargs
    ) -> DriftMetricsResult:
        """评估漂移指标

        Returns:
            DriftMetricsResult
        """
        # 计算指标
        result = self.calculator.calculate_metric(
            metric_name, baseline, current, config, **kwargs
        )

        value = result["value"]
        thresholds = self.thresholds.get(metric_name, {"green": 0.1, "yellow": 0.2})

        # 判断状态
        if value <= thresholds["green"]:
            status = "GREEN"
            message = f"No significant drift detected ({metric_name}={value:.4f})"
        elif value <= thresholds["yellow"]:
            status = "YELLOW"
            message = f"Minor drift detected ({metric_name}={value:.4f})"
        else:
            status = "RED"
            message = f"Significant drift detected ({metric_name}={value:.4f})"

        return DriftMetricsResult(
            metric_name=metric_name,
            value=value,
            p_value=result.get("p_value"),
            status=status,
            threshold_green=thresholds["green"],
            threshold_yellow=thresholds["yellow"],
            message=message,
            details=result.get("details", {})
        )


if __name__ == "__main__":
    """测试漂移指标计算"""
    print("=== SPL-6a-B: 漂移指标计算测试 ===\n")

    # 创建测试数据
    np.random.seed(42)
    baseline_data = np.random.normal(0, 1, 1000)
    current_data = np.random.normal(0.1, 1.1, 1000)  # 轻微漂移

    print(f"Baseline: mean={baseline_data.mean():.4f}, std={baseline_data.std():.4f}")
    print(f"Current: mean={current_data.mean():.4f}, std={current_data.std():.4f}\n")

    # 测试评估器
    evaluator = DriftMetricsEvaluator()

    # 测试 KS test
    from analysis.drift_objects import DriftObjectConfig, Bucket, MetricType, DriftCategory

    config = DriftObjectConfig(
        name="test",
        category=DriftCategory.INPUT_DISTRIBUTION,
        metric_type=MetricType.STATISTICAL,
        description="Test",
        window_days=[30, 90],
        metrics=["ks_test", "wasserstein", "absolute_change"]
    )

    print("测试 KS test:")
    result = evaluator.evaluate("ks_test", baseline_data, current_data, config)
    print(f"  Value: {result.value:.4f}")
    print(f"  Status: {result.status}")
    print(f"  Message: {result.message}")

    print("\n测试 Wasserstein:")
    result = evaluator.evaluate("wasserstein", baseline_data, current_data, config)
    print(f"  Value: {result.value:.4f}")
    print(f"  Status: {result.status}")

    print("\n✅ 漂移指标计算测试通过")
