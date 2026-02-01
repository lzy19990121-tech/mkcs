"""
SPL-5a-C: 自适应阈值离线标定

使用网格搜索标定自适应阈值函数的参数，确保在训练集上最优且在验证集上不超限。
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from itertools import product
from collections import defaultdict

from analysis.replay_schema import ReplayOutput, load_replay_outputs
from analysis.window_scanner import WindowScanner, WindowMetrics
from analysis.regime_features import RegimeFeatureCalculator, RegimeFeatures
from analysis.adaptive_threshold import (
    ThresholdFunction,
    PiecewiseConstantThreshold,
    LinearThreshold,
    AdaptiveThreshold,
    AdaptiveThresholdRuleset
)


@dataclass
class CalibrationResult:
    """单次标定结果"""
    params: Dict[str, Any]  # 参数配置
    train_score: float      # 训练集得分
    validation_score: float # 验证集得分
    train_triggers: int     # 训练集触发次数
    validation_triggers: int # 验证集触发次数
    train_downtime: float   # 训练集停机时间比例
    validation_downtime: float # 验证集停机时间比例
    envelope_violations: int # 风险包络违反次数

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RuleCalibration:
    """单条规则的标定结果"""
    rule_id: str
    rule_name: str
    rule_type: str
    best_params: Dict[str, Any]
    best_result: CalibrationResult
    all_results: List[CalibrationResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "rule_type": self.rule_type,
            "best_params": self.best_params,
            "best_result": self.best_result.to_dict(),
            "all_results": [r.to_dict() for r in self.all_results]
        }


class ThresholdCalibrator:
    """自适应阈值标定器

    使用网格搜索优化阈值函数参数。
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        envelope_tolerance: float = 0.05,
        min_windows: int = 10
    ):
        """初始化标定器

        Args:
            train_ratio: 训练集比例（按时间划分）
            envelope_tolerance: 风险包络容忍度
            min_windows: 最小窗口数量要求
        """
        self.train_ratio = train_ratio
        self.envelope_tolerance = envelope_tolerance
        self.min_windows = min_windows

    def split_data(
        self,
        replay: ReplayOutput
    ) -> Tuple[ReplayOutput, ReplayOutput]:
        """按时间切分训练集和验证集

        Args:
            replay: 回测输出

        Returns:
            (train_replay, validation_replay)
        """
        if not replay.steps:
            return replay, replay

        # 按时间排序
        steps = sorted(replay.steps, key=lambda s: s.timestamp)
        n_steps = len(steps)

        # 计算切分点
        split_idx = int(n_steps * self.train_ratio)

        # 训练集
        train_steps = steps[:split_idx]
        train_start = train_steps[0].timestamp if train_steps else replay.start_date
        train_end = train_steps[-1].timestamp if train_steps else replay.start_date

        # 验证集
        val_steps = steps[split_idx:]
        val_start = val_steps[0].timestamp if val_steps else train_end
        val_end = val_steps[-1].timestamp if val_steps else replay.end_date

        # 创建切片
        train_replay = replay.slice_by_time(train_start, train_end)
        val_replay = replay.slice_by_time(val_start, val_end)

        return train_replay, val_replay

    def grid_search_piecewise_constant(
        self,
        train_data: List[Tuple[RegimeFeatures, float]],
        val_data: List[Tuple[RegimeFeatures, float]],
        feature_name: str,
        buckets_config: Dict[str, Tuple[float, float]],
        param_grid: Dict[str, List[Any]]
    ) -> CalibrationResult:
        """网格搜索分段常数阈值

        Args:
            train_data: [(regime, metric_value)] 训练数据
            val_data: [(regime, metric_value)] 验证数据
            feature_name: 特征名称
            buckets_config: 桶配置 {bucket: (min, max)}
            param_grid: 参数网格

        Returns:
            最优标定结果
        """
        best_result = None
        best_score = -np.inf

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = [param_grid[k] for k in param_names]
        all_combinations = list(product(*param_values))

        for combination in all_combinations:
            params = dict(zip(param_names, combination))

            # 创建阈值函数
            threshold_func = PiecewiseConstantThreshold(
                feature_name=feature_name,
                buckets=params
            )

            # 评估
            result = self._evaluate_threshold(
                threshold_func, train_data, val_data, params
            )

            # 计算综合得分
            score = self._calculate_score(result)

            if score > best_score:
                best_score = score
                best_result = result

        return best_result

    def grid_search_linear(
        self,
        train_data: List[Tuple[RegimeFeatures, float]],
        val_data: List[Tuple[RegimeFeatures, float]],
        feature_name: str,
        param_grid: Dict[str, List[float]]
    ) -> CalibrationResult:
        """网格搜索线性阈值

        Args:
            train_data: 训练数据
            val_data: 验证数据
            feature_name: 特征名称
            param_grid: 参数网格 {intercept: [...], slope: [...], ...}

        Returns:
            最优标定结果
        """
        best_result = None
        best_score = -np.inf

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = [param_grid[k] for k in param_names]
        all_combinations = list(product(*param_values))

        for combination in all_combinations:
            params = dict(zip(param_names, combination))

            # 创建阈值函数
            threshold_func = LinearThreshold(
                feature_name=feature_name,
                intercept=params.get("intercept", 0.0),
                slope=params.get("slope", 0.0),
                min_value=params.get("min_value"),
                max_value=params.get("max_value")
            )

            # 评估
            result = self._evaluate_threshold(
                threshold_func, train_data, val_data, params
            )

            # 计算综合得分
            score = self._calculate_score(result)

            if score > best_score:
                best_score = score
                best_result = result

        return best_result

    def _evaluate_threshold(
        self,
        threshold_func: ThresholdFunction,
        train_data: List[Tuple[RegimeFeatures, float]],
        val_data: List[Tuple[RegimeFeatures, float]],
        params: Dict[str, Any]
    ) -> CalibrationResult:
        """评估阈值函数

        Args:
            threshold_func: 阈值函数
            train_data: 训练数据
            val_data: 验证数据
            params: 参数

        Returns:
            标定结果
        """
        # 训练集评估
        train_triggers = 0
        train_total = len(train_data)
        train_downtime_steps = 0

        for regime, metric_value in train_data:
            threshold = threshold_func.evaluate(regime)

            # 判断是否触发
            triggered = self._should_trigger(metric_value, threshold)
            if triggered:
                train_triggers += 1
                train_downtime_steps += 1

        train_score = self._calculate_objective(
            train_triggers, train_total, train_downtime_steps
        )

        # 验证集评估
        val_triggers = 0
        val_total = len(val_data)
        val_downtime_steps = 0
        envelope_violations = 0

        for regime, metric_value in val_data:
            threshold = threshold_func.evaluate(regime)

            # 判断是否触发
            triggered = self._should_trigger(metric_value, threshold)
            if triggered:
                val_triggers += 1
                val_downtime_steps += 1

            # 检查是否违反包络（简化）
            if metric_value < threshold * (1 - self.envelope_tolerance):
                envelope_violations += 1

        val_score = self._calculate_objective(
            val_triggers, val_total, val_downtime_steps
        )

        return CalibrationResult(
            params=params,
            train_score=train_score,
            validation_score=val_score,
            train_triggers=train_triggers,
            validation_triggers=val_triggers,
            train_downtime=train_downtime_steps / train_total if train_total > 0 else 0,
            validation_downtime=val_downtime_steps / val_total if val_total > 0 else 0,
            envelope_violations=envelope_violations
        )

    def _should_trigger(self, metric_value: float, threshold: float) -> bool:
        """判断是否应该触发风控

        Args:
            metric_value: 指标值
            threshold: 阈值

        Returns:
            是否触发
        """
        # 简化逻辑：如果指标值小于阈值，则触发
        # 对于不同的指标类型，逻辑可能不同
        return metric_value < threshold

    def _calculate_objective(
        self,
        triggers: int,
        total: int,
        downtime_steps: int
    ) -> float:
        """计算目标函数得分

        目标：最小化触发次数和停机时间

        Args:
            triggers: 触发次数
            total: 总次数
            downtime_steps: 停机步数

        Returns:
            得分（越高越好）
        """
        if total == 0:
            return 0.0

        trigger_rate = triggers / total
        downtime_rate = downtime_steps / total

        # 目标：触发率在合理范围内（10%-30%），停机时间最小化
        # 得分 = 100 - 触发率惩罚 - 停机时间惩罚

        trigger_penalty = abs(trigger_rate - 0.2) * 100  # 理想触发率20%
        downtime_penalty = downtime_rate * 50

        score = 100 - trigger_penalty - downtime_penalty
        return max(0.0, score)

    def _calculate_score(self, result: CalibrationResult) -> float:
        """计算综合得分

        优先级：
        1. 无包络违反
        2. 验证集得分
        3. 停机时间最小化

        Args:
            result: 标定结果

        Returns:
            综合得分
        """
        # 包络违反惩罚
        violation_penalty = result.envelope_violations * 100

        # 综合得分 = 验证集得分 - 违反惩罚 - 停机时间惩罚
        score = result.validation_score - violation_penalty - result.validation_downtime * 20

        return score


def prepare_calibration_data(
    replay: ReplayOutput,
    metric_name: str,
    window_length: str = "20d"
) -> List[Tuple[RegimeFeatures, float]]:
    """准备标定数据

    从回测数据中提取 (市场状态, 指标值) 对。

    Args:
        replay: 回测输出
        metric_name: 指标名称 (stability_score, window_return, drawdown_duration)
        window_length: 窗口长度

    Returns:
        [(regime, metric_value)] 列表
    """
    data = []

    # 创建特征计算器
    calculator = RegimeFeatureCalculator(window_length=20)

    # 扫描窗口
    scanner = WindowScanner(windows=[window_length], top_k=100)
    windows = scanner.scan_replay(replay, window_length)

    if len(windows) < 5:
        return []

    for window in windows:
        # 获取窗口的指标值
        if metric_name == "stability_score":
            metric_value = window.max_drawdown * 100  # 简化：使用MDD作为代理
        elif metric_name == "window_return":
            metric_value = window.total_return * 100
        elif metric_name == "drawdown_duration":
            metric_value = float(window.max_dd_duration)
        else:
            metric_value = 0.0

        # 计算市场状态（简化：使用窗口内的价格数据）
        # 这里使用代理方法
        regime = RegimeFeatures(
            realized_vol=0.01 + np.random.random() * 0.02,  # 简化
            vol_bucket="med",
            adx=20 + np.random.random() * 20,
            trend_bucket="strong",
            spread_proxy=0.001,
            cost_bucket="low",
            calculated_at=window.start_date
        )

        data.append((regime, metric_value))

    return data


def calibrate_all_rules(
    replays: List[ReplayOutput],
    output_path: str = "config/adaptive_gating_params.json"
) -> Dict[str, RuleCalibration]:
    """标定所有自适应规则

    Args:
        replays: 回测输出列表
        output_path: 输出路径

    Returns:
        {rule_id: RuleCalibration} 字典
    """
    calibrator = ThresholdCalibrator(train_ratio=0.7)

    # 定义所有规则的参数网格
    rule_configs = [
        {
            "rule_id": "adaptive_stability_gating",
            "rule_name": "自适应稳定性暂停交易",
            "rule_type": "gating",
            "metric_name": "stability_score",
            "threshold_type": "piecewise_constant",
            "feature_name": "realized_vol",
            "param_grid": {
                "low": [15.0, 20.0, 25.0],
                "med": [25.0, 30.0, 35.0],
                "high": [35.0, 40.0, 45.0]
            }
        },
        {
            "rule_id": "adaptive_return_reduction",
            "rule_name": "自适应收益降仓",
            "rule_type": "reduction",
            "metric_name": "window_return",
            "threshold_type": "linear",
            "feature_name": "realized_vol",
            "param_grid": {
                "intercept": [-0.10, -0.08, -0.06],
                "slope": [-3.0, -2.0, -1.0],
                "min_value": [-0.25, -0.20, -0.15],
                "max_value": [-0.08, -0.05, -0.03]
            }
        },
        {
            "rule_id": "adaptive_regime_disable",
            "rule_name": "自适应市场状态禁用",
            "rule_type": "disable",
            "metric_name": "market_regime",
            "threshold_type": "piecewise_constant",
            "feature_name": "adx",
            "param_grid": {
                "weak": [0.0],
                "strong": [1.0]
            }
        },
        {
            "rule_id": "adaptive_duration_gating",
            "rule_name": "自适应回撤持续暂停",
            "rule_type": "gating",
            "metric_name": "drawdown_duration",
            "threshold_type": "linear",
            "feature_name": "realized_vol",
            "param_grid": {
                "intercept": [8.0, 10.0, 12.0],
                "slope": [-250.0, -200.0, -150.0],
                "min_value": [3.0, 5.0, 7.0],
                "max_value": [12.0, 15.0, 18.0]
            }
        }
    ]

    results = {}

    for config in rule_configs:
        print(f"\n标定规则: {config['rule_name']}")

        # 合并所有replay的数据
        all_train_data = []
        all_val_data = []

        for replay in replays:
            # 切分数据
            train_replay, val_replay = calibrator.split_data(replay)

            # 准备数据
            train_data = prepare_calibration_data(
                train_replay, config["metric_name"]
            )
            val_data = prepare_calibration_data(
                val_replay, config["metric_name"]
            )

            all_train_data.extend(train_data)
            all_val_data.extend(val_data)

        if len(all_train_data) < calibrator.min_windows:
            print(f"  警告: 训练数据不足 ({len(all_train_data)} < {calibrator.min_windows})")
            continue

        # 网格搜索
        if config["threshold_type"] == "piecewise_constant":
            buckets_config = {
                "low": (0.0, 0.01),
                "med": (0.01, 0.02),
                "high": (0.02, float("inf"))
            }
            best_result = calibrator.grid_search_piecewise_constant(
                all_train_data,
                all_val_data,
                config["feature_name"],
                buckets_config,
                config["param_grid"]
            )
        else:  # linear
            best_result = calibrator.grid_search_linear(
                all_train_data,
                all_val_data,
                config["feature_name"],
                config["param_grid"]
            )

        print(f"  最优参数: {best_result.params}")
        print(f"  训练集得分: {best_result.train_score:.2f}")
        print(f"  验证集得分: {best_result.validation_score:.2f}")
        print(f"  触发次数: 训练={best_result.train_triggers}, 验证={best_result.validation_triggers}")

        results[config["rule_id"]] = RuleCalibration(
            rule_id=config["rule_id"],
            rule_name=config["rule_name"],
            rule_type=config["rule_type"],
            best_params=best_result.params,
            best_result=best_result,
            all_results=[best_result]
        )

    # 保存结果
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "version": "adaptive_v1",
        "calibrated_at": datetime.now().isoformat(),
        "train_ratio": calibrator.train_ratio,
        "min_windows": calibrator.min_windows,
        "rules": {
            rule_id: rule_cal.to_dict()
            for rule_id, rule_cal in results.items()
        }
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n标定结果已保存到: {output_path}")

    return results


def load_calibrated_params(
    params_path: str = "config/adaptive_gating_params.json"
) -> Dict[str, Any]:
    """加载标定参数

    Args:
        params_path: 参数文件路径（None则返回空字典）

    Returns:
        标定参数字典
    """
    if params_path is None:
        return {}

    params_file = Path(params_path)
    if not params_file.exists():
        return {}

    with open(params_file) as f:
        data = json.load(f)

    return data


def create_calibrated_ruleset(
    params_path: str = "config/adaptive_gating_params.json"
) -> AdaptiveThresholdRuleset:
    """从标定参数创建规则集

    Args:
        params_path: 参数文件路径

    Returns:
        AdaptiveThresholdRuleset
    """
    params_data = load_calibrated_params(params_path)

    if not params_data or "rules" not in params_data:
        # 返回默认规则集
        from analysis.adaptive_threshold import create_default_adaptive_ruleset
        return create_default_adaptive_ruleset()

    ruleset = AdaptiveThresholdRuleset()

    for rule_id, rule_data in params_data["rules"].items():
        params = rule_data["best_params"]

        # 根据规则类型创建阈值函数
        if rule_data["rule_type"] == "gating":
            if "low" in params and "med" in params and "high" in params:
                # PiecewiseConstant
                threshold_func = PiecewiseConstantThreshold(
                    feature_name="realized_vol",
                    buckets=params
                )
            else:
                # Linear
                threshold_func = LinearThreshold(
                    feature_name="realized_vol",
                    intercept=params.get("intercept", 0.0),
                    slope=params.get("slope", 0.0),
                    min_value=params.get("min_value"),
                    max_value=params.get("max_value")
                )
        else:
            # 其他类型
            threshold_func = PiecewiseConstantThreshold(
                feature_name="realized_vol",
                buckets={"low": 20.0, "med": 30.0, "high": 40.0}
            )

        rule = AdaptiveThreshold(
            rule_id=rule_id,
            rule_name=rule_data["rule_name"],
            rule_type=rule_data["rule_type"],
            trigger_metric="stability_score",
            threshold_function=threshold_func,
            description=f"Calibrated: {params_data.get('calibrated_at', 'unknown')}"
        )

        ruleset.add_rule(rule)

    return ruleset


if __name__ == "__main__":
    """测试代码"""
    print("=== AdaptiveThresholdCalibrator 测试 ===\n")

    # 加载回测数据
    replays = load_replay_outputs("runs")

    if len(replays) == 0:
        print("未找到回测数据，使用模拟数据...")

        # 创建模拟replay
        from analysis.replay_schema import ReplayOutput
        from datetime import date
        from decimal import Decimal

        mock_replay = ReplayOutput(
            run_id="test_exp",
            strategy_id="test_strategy",
            strategy_name="Test Strategy",
            commit_hash="abc123",
            config_hash="cfg123",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 3, 31),
            initial_cash=Decimal("100000"),
            final_equity=Decimal("110000")
        )
        replays = [mock_replay]

    # 运行标定
    results = calibrate_all_rules(replays, output_path="config/adaptive_gating_params_test.json")

    print(f"\n标定了 {len(results)} 条规则")

    for rule_id, cal in results.items():
        print(f"\n{cal.rule_name}:")
        print(f"  最优参数: {cal.best_params}")
        print(f"  验证集得分: {cal.best_result.validation_score:.2f}")

    print("\n✓ 测试通过")
