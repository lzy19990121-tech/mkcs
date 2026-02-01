"""
SPL-6a-C: 漂移阈值与分级响应

实现漂移分级响应策略（GREEN/YELLOW/RED）和再标定触发条件。
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import yaml
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.drift_objects import DriftObjectConfig, DriftResult, DriftCategory


class DriftStatus(Enum):
    """漂移状态"""
    GREEN = "GREEN"       # 无显著漂移
    YELLOW = "YELLOW"     # 轻微漂移
    RED = "RED"           # 显著漂移
    CRITICAL = "CRITICAL" # 严重漂移


@dataclass
class ThresholdConfig:
    """阈值配置"""
    green: float
    yellow: float
    critical: Optional[float] = None

    def get_status(self, value: float) -> DriftStatus:
        """根据值判断状态"""
        if self.critical and value >= self.critical:
            return DriftStatus.CRITICAL
        if value >= self.yellow:
            return DriftStatus.RED
        if value >= self.green:
            return DriftStatus.YELLOW
        return DriftStatus.GREEN


@dataclass
class RecalibrationTrigger:
    """再标定触发条件"""
    triggered: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class DriftThresholdEvaluator:
    """漂移阈值评估器"""

    def __init__(self, config_path: Optional[Path] = None):
        """初始化评估器

        Args:
            config_path: 阈值配置文件路径
        """
        self.config_path = config_path or Path("config/drift_thresholds.yaml")
        self.thresholds = self._load_thresholds()
        self.response_policy = self._load_response_policy()
        self.recalibration_config = self._load_recalibration_config()
        self.priority_levels = self._load_priority_levels()

    def _load_thresholds(self) -> Dict:
        """加载阈值配置"""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        return config.get("thresholds", {})

    def _load_response_policy(self) -> Dict:
        """加载响应策略"""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        return config.get("response_policy", {})

    def _load_recalibration_config(self) -> Dict:
        """加载再标定触发配置"""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        return config.get("recalibration_trigger", {})

    def _load_priority_levels(self) -> Dict:
        """加载优先级配置"""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        return config.get("priority_levels", {})

    def get_threshold(self, object_name: str, metric_name: str) -> ThresholdConfig:
        """获取对象的阈值配置

        Args:
            object_name: 漂移对象名（如 "input_distribution.returns"）
            metric_name: 指标名（如 "psi"）

        Returns:
            ThresholdConfig
        """
        # 解析类别
        category = object_name.split(".")[0]

        # 查找阈值
        category_thresholds = self.thresholds.get(category, {})
        metric_thresholds = category_thresholds.get(metric_name, {})

        # 默认阈值
        default_green = metric_thresholds.get("green", 0.1)
        default_yellow = metric_thresholds.get("yellow", 0.2)
        default_critical = metric_thresholds.get("critical")

        return ThresholdConfig(
            green=default_green,
            yellow=default_yellow,
            critical=default_critical
        )

    def evaluate_drift_result(
        self,
        result: DriftResult,
        config: DriftObjectConfig
    ) -> DriftResult:
        """评估漂移结果，添加状态和消息

        Args:
            result: 漂移检测结果
            config: 漂移对象配置

        Returns:
            更新后的 DriftResult
        """
        # 获取阈值
        threshold_config = self.get_threshold(
            f"{config.category.value}.{config.name}",
            result.metric_name
        )

        # 判断状态
        status = threshold_config.get_status(result.drift_value)
        result.status = status.value
        result.threshold = threshold_config.yellow

        # 生成消息
        response = self.response_policy.get(status.value, {})
        result.message = response.get("message", f"Drift detected: {result.metric_name}={result.drift_value:.4f}")

        return result

    def check_recalibration_trigger(
        self,
        drift_results: List[DriftResult],
        historical_red_count: Dict[str, int],
        last_recalibration_date: Optional[datetime] = None
    ) -> RecalibrationTrigger:
        """检查是否触发再标定

        Args:
            drift_results: 所有漂移检测结果
            historical_red_count: 历史红次数统计
            last_recalibration_date: 上次再标定日期

        Returns:
            RecalibrationTrigger
        """
        trigger_config = self.recalibration_config
        conditions = trigger_config.get("conditions", {})
        protections = trigger_config.get("protections", {})

        reasons = []

        # 检查保护机制
        if last_recalibration_date:
            cooldown_days = protections.get("cooldown_after_recalibration", 60)
            days_since_recal = (datetime.now() - last_recalibration_date).days
            if days_since_recal < cooldown_days:
                return RecalibrationTrigger(
                    triggered=False,
                    reason=f"在冷却期内（{days_since_recal}/{cooldown_days}天）",
                    details={"cooldown_remaining": cooldown_days - days_since_recal}
                )

        # 条件1: 连续 RED 检测
        if conditions.get("consecutive_red", {}).get("enabled"):
            config = conditions["consecutive_red"]
            required_count = config.get("count", 3)
            window_days = config.get("window_days", 7)

            # 统计最近的 RED 次数
            recent_red_count = sum(
                1 for r in drift_results
                if r.status == "RED" and
                (datetime.now() - r.current_snapshot.timestamp).days <= window_days
            )

            if recent_red_count >= required_count:
                reasons.append(f"连续 {recent_red_count} 次 RED 检测（要求 {required_count} 次）")

        # 条件2: 关键风险指标退化
        if conditions.get("key_risk_degradation", {}).get("enabled"):
            config = conditions["key_risk_degradation"]
            key_metrics = config.get("metrics", [])

            for metric_config in key_metrics:
                metric_name = metric_config["name"]
                threshold = metric_config["threshold"]

                # 查找对应的结果
                for result in drift_results:
                    if result.object_name == metric_name:
                        if result.drift_value > threshold:
                            reasons.append(f"关键风险指标 {metric_name} 恶化 {result.drift_value:.2%}（阈值 {threshold:.2%}）")

        # 条件3: 多个对象同时 YELLOW
        if conditions.get("multiple_yellow", {}).get("enabled"):
            config = conditions["multiple_yellow"]
            required_count = config.get("count", 5)
            required_ratio = config.get("ratio", 0.3)

            yellow_count = sum(1 for r in drift_results if r.status == "YELLOW")
            total_count = len(drift_results)
            yellow_ratio = yellow_count / total_count if total_count > 0 else 0

            if yellow_count >= required_count or yellow_ratio >= required_ratio:
                reasons.append(f"多个对象同时 YELLOW：{yellow_count}/{total_count}（要求 {required_count} 个或 {required_ratio:.0%}）")

        # 条件4: 组合协同爆炸
        if conditions.get("portfolio_co_crash", {}).get("enabled"):
            config = conditions["portfolio_co_crash"]
            required_co_crash = config.get("co_crash_count", 3)
            required_frequency = config.get("frequency", 2)

            # 查找 co_crash_frequency 结果
            for result in drift_results:
                if "co_crash_frequency" in result.object_name:
                    if result.drift_value >= required_frequency:
                        reasons.append(f"组合协同爆炸：{result.drift_value} 次（阈值 {required_frequency}）")

        # 判断是否触发
        triggered = len(reasons) > 0

        return RecalibrationTrigger(
            triggered=triggered,
            reason="; ".join(reasons) if triggered else "无触发条件满足",
            details={
                "trigger_count": len(reasons),
                "reasons": reasons
            }
        )

    def get_response_action(self, status: DriftStatus) -> Dict[str, Any]:
        """获取响应行动

        Args:
            status: 漂移状态

        Returns:
            响应行动配置
        """
        policy = self.response_policy.get(status.value, {})
        return {
            "action": policy.get("action", "none"),
            "message": policy.get("message", ""),
            "monitoring": policy.get("monitoring", "normal"),
            "notifications": policy.get("notifications", []),
            "triggers": policy.get("triggers", [])
        }

    def get_object_priority(self, object_name: str) -> str:
        """获取对象优先级

        Args:
            object_name: 漂移对象名

        Returns:
            优先级：critical/high/medium/low
        """
        for priority, objects in self.priority_levels.items():
            if object_name in objects:
                return priority
        return "low"


class DriftReportGenerator:
    """漂移报告生成器"""

    def __init__(self, evaluator: DriftThresholdEvaluator):
        """初始化报告生成器

        Args:
            evaluator: 阈值评估器
        """
        self.evaluator = evaluator

    def generate_summary_report(
        self,
        drift_results: List[DriftResult],
        recalibration_trigger: RecalibrationTrigger
    ) -> Dict[str, Any]:
        """生成汇总报告

        Args:
            drift_results: 所有漂移检测结果
            recalibration_trigger: 再标定触发结果

        Returns:
            汇总报告字典
        """
        # 统计各状态数量
        status_counts = {status.value: 0 for status in DriftStatus}
        for result in drift_results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1

        # 按优先级分组
        priority_breakdown = {
            "critical": {"GREEN": 0, "YELLOW": 0, "RED": 0, "CRITICAL": 0},
            "high": {"GREEN": 0, "YELLOW": 0, "RED": 0, "CRITICAL": 0},
            "medium": {"GREEN": 0, "YELLOW": 0, "RED": 0, "CRITICAL": 0},
            "low": {"GREEN": 0, "YELLOW": 0, "RED": 0, "CRITICAL": 0}
        }

        for result in drift_results:
            priority = self.evaluator.get_object_priority(result.object_name)
            priority_breakdown[priority][result.status] += 1

        # 关键漂移
        critical_drifts = [
            r for r in drift_results
            if r.status in ["RED", "CRITICAL"]
        ]

        return {
            "report_date": datetime.now().isoformat(),
            "total_objects": len(drift_results),
            "status_counts": status_counts,
            "priority_breakdown": priority_breakdown,
            "critical_drifts": [
                {
                    "object": r.object_name,
                    "metric": r.metric_name,
                    "value": r.drift_value,
                    "status": r.status
                }
                for r in critical_drifts
            ],
            "recalibration_triggered": recalibration_trigger.triggered,
            "recalibration_reason": recalibration_trigger.reason,
            "overall_status": self._get_overall_status(status_counts)
        }

    def _get_overall_status(self, status_counts: Dict[str, int]) -> str:
        """获取总体状态"""
        if status_counts.get("CRITICAL", 0) > 0:
            return "CRITICAL"
        if status_counts.get("RED", 0) > 0:
            return "RED"
        if status_counts.get("YELLOW", 0) > 0:
            return "YELLOW"
        return "GREEN"


if __name__ == "__main__":
    """测试漂移阈值评估"""
    print("=== SPL-6a-C: 漂移阈值与分级响应测试 ===\n")

    # 创建评估器
    evaluator = DriftThresholdEvaluator()

    # 测试阈值配置
    print("测试阈值配置:")
    threshold = evaluator.get_threshold("input_distribution.returns", "psi")
    print(f"  PSI thresholds: GREEN={threshold.green}, YELLOW={threshold.yellow}")
    print(f"  测试值 0.05 -> {threshold.get_status(0.05).value}")
    print(f"  测试值 0.15 -> {threshold.get_status(0.15).value}")
    print(f"  测试值 0.30 -> {threshold.get_status(0.30).value}")

    # 测试优先级
    print("\n测试对象优先级:")
    critical_objects = evaluator.priority_levels.get("critical", [])
    print(f"  关键对象数量: {len(critical_objects)}")
    print(f"  示例: {critical_objects[:3]}")

    # 测试响应策略
    print("\n测试响应策略:")
    for status in [DriftStatus.GREEN, DriftStatus.YELLOW, DriftStatus.RED]:
        action = evaluator.get_response_action(status)
        print(f"  {status.value}: action={action['action']}, monitoring={action['monitoring']}")

    print("\n✅ 漂移阈值与分级响应测试通过")
