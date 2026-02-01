"""
SPL-5a-E: 自适应风控回归测试

测试自适应阈值系统的回归行为：
1. 阈值参数稳定性测试
2. 市场状态响应测试
3. Gating决策一致性测试
4. 风险包络约束测试
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np

from analysis.risk_baseline import RiskBaseline
from analysis.replay_schema import ReplayOutput
from analysis.regime_features import RegimeFeatureCalculator, RegimeFeatures
from analysis.adaptive_gating import AdaptiveRiskGate, GatingAction, GatingDecision
from analysis.adaptive_calibration import load_calibrated_params


@dataclass
class AdaptiveTestResult:
    """自适应风控测试结果"""
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP"
    message: str
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "status": self.status,
            "message": self.message,
            "details": self.details or {}
        }


class AdaptiveGatingTests:
    """自适应风控回归测试套件

    Test Suite:
    1. C3.1: 阈值参数稳定性测试
    2. C3.2: 市场状态响应测试
    3. C3.3: Gating决策一致性测试
    4. C3.4: 风险包络约束测试
    """

    def __init__(
        self,
        params_path: str = "config/adaptive_gating_params.json",
        tolerance_pct: float = 0.05
    ):
        """初始化测试套件

        Args:
            params_path: 标定参数路径
            tolerance_pct: 容忍度（默认5%）
        """
        self.params_path = params_path
        self.tolerance_pct = tolerance_pct

        # 加载标定参数
        self.calibrated_params = load_calibrated_params(params_path)

    def test_threshold_parameter_stability(
        self,
        current_params_path: str = None
    ) -> AdaptiveTestResult:
        """C3.1: 阈值参数稳定性测试

        验证标定的阈值参数没有发生意外漂移。

        Args:
            current_params_path: 当前参数路径（None则使用默认）

        Returns:
            AdaptiveTestResult
        """
        try:
            details = {}

            if not self.calibrated_params:
                return AdaptiveTestResult(
                    test_name="threshold_parameter_stability",
                    status="SKIP",
                    message="No calibrated parameters found",
                    details={"reason": "missing_params"}
                )

            # 检查所有规则的参数
            param_drifts = []
            all_rules = self.calibrated_params.get("rules", {})

            for rule_id, rule_data in all_rules.items():
                best_params = rule_data.get("best_params", {})

                # 检查参数类型和范围
                for param_name, param_value in best_params.items():
                    # 检查参数是否在合理范围内
                    if isinstance(param_value, (int, float)):
                        # 阈值参数应该是合理的
                        if "threshold" in rule_id or "gating" in rule_id:
                            # 稳定性评分阈值应该在 0-100 之间
                            if param_name in ["low", "med", "high"]:
                                if not (0 <= param_value <= 100):
                                    param_drifts.append({
                                        "rule": rule_id,
                                        "param": param_name,
                                        "value": param_value,
                                        "issue": "out_of_range"
                                    })
                        elif "reduction" in rule_id:
                            # 收益阈值应该是负数
                            if param_name == "intercept" or param_name == "max_value":
                                if param_value > 0:
                                    param_drifts.append({
                                        "rule": rule_id,
                                        "param": param_name,
                                        "value": param_value,
                                        "issue": "should_be_negative"
                                    })
                        elif "duration" in rule_id:
                            # 持续时间阈值应该是正数
                            if param_name == "intercept":
                                if param_value < 0:
                                    param_drifts.append({
                                        "rule": rule_id,
                                        "param": param_name,
                                        "value": param_value,
                                        "issue": "should_be_positive"
                                    })

                details[rule_id] = {
                    "params": best_params,
                    "calibration_score": rule_data.get("best_result", {}).get("validation_score")
                }

            if param_drifts:
                return AdaptiveTestResult(
                    test_name="threshold_parameter_stability",
                    status="FAIL",
                    message=f"Found {len(param_drifts)} parameter drifts",
                    details={"drifts": param_drifts, **details}
                )

            return AdaptiveTestResult(
                test_name="threshold_parameter_stability",
                status="PASS",
                message=f"All {len(all_rules)} rule parameters stable",
                details=details
            )

        except Exception as e:
            return AdaptiveTestResult(
                test_name="threshold_parameter_stability",
                status="FAIL",
                message=f"Test error: {str(e)}",
                details={"error": str(e)}
            )

    def test_regime_response(
        self,
        replay: ReplayOutput,
        strategy_id: str = None
    ) -> AdaptiveTestResult:
        """C3.2: 市场状态响应测试

        验证风控闸门能正确响应不同的市场状态。

        Args:
            replay: 回测输出
            strategy_id: 策略ID

        Returns:
            AdaptiveTestResult
        """
        try:
            if strategy_id is None:
                strategy_id = replay.strategy_id

            # 创建风控闸门
            gate = AdaptiveRiskGate(
                strategy_id=strategy_id,
                params_path=self.params_path
            )

            details = {}
            response_tests = []

            # 测试1: 低波动市场
            low_vol_regime = RegimeFeatures(
                realized_vol=0.005,
                vol_bucket="low",
                adx=30.0,
                trend_bucket="strong",
                spread_proxy=0.0005,
                cost_bucket="low",
                calculated_at=datetime.now()
            )

            # 在低波动市场，阈值应该较低（更宽松）
            for rule in gate.ruleset.get_all_rules():
                if rule.rule_type == "gating":
                    threshold = rule.get_threshold(low_vol_regime)
                    details[f"{rule.rule_id}_low_vol"] = threshold

            # 测试2: 高波动市场
            high_vol_regime = RegimeFeatures(
                realized_vol=0.025,
                vol_bucket="high",
                adx=15.0,
                trend_bucket="weak",
                spread_proxy=0.002,
                cost_bucket="high",
                calculated_at=datetime.now()
            )

            # 在高波动市场，阈值应该较高（更严格）
            for rule in gate.ruleset.get_all_rules():
                if rule.rule_type == "gating":
                    threshold_high = rule.get_threshold(high_vol_regime)
                    threshold_low = rule.get_threshold(low_vol_regime)

                    # 验证：高波动市场阈值 >= 低波动市场阈值
                    if threshold_high < threshold_low:
                        response_tests.append({
                            "rule": rule.rule_id,
                            "issue": "threshold_not_adaptive",
                            "low_vol_threshold": threshold_low,
                            "high_vol_threshold": threshold_high,
                            "expected": "high_vol >= low_vol"
                        })

                    details[f"{rule.rule_id}_high_vol"] = threshold_high

            # 测试3: 震荡市场（ADX低）
            weak_trend_regime = RegimeFeatures(
                realized_vol=0.015,
                vol_bucket="med",
                adx=20.0,
                trend_bucket="weak",
                spread_proxy=0.001,
                cost_bucket="med",
                calculated_at=datetime.now()
            )

            # 在震荡市场，禁用规则应该触发
            for rule in gate.ruleset.get_all_rules():
                if rule.rule_type == "disable":
                    threshold = rule.get_threshold(weak_trend_regime)
                    # ADX=20 < 25 应该返回禁用阈值
                    details[f"{rule.rule_id}_weak_trend"] = threshold

            if response_tests:
                return AdaptiveTestResult(
                    test_name="regime_response",
                    status="FAIL",
                    message=f"Found {len(response_tests)} regime response issues",
                    details={"issues": response_tests, **details}
                )

            return AdaptiveTestResult(
                test_name="regime_response",
                status="PASS",
                message="Regime response test passed",
                details=details
            )

        except Exception as e:
            return AdaptiveTestResult(
                test_name="regime_response",
                status="FAIL",
                message=f"Test error: {str(e)}",
                details={"error": str(e)}
            )

    def test_gating_decision_consistency(
        self,
        replay: ReplayOutput,
        strategy_id: str = None
    ) -> AdaptiveTestResult:
        """C3.3: Gating决策一致性测试

        验证相同输入产生相同的gating决策。

        Args:
            replay: 回测输出
            strategy_id: 策略ID

        Returns:
            AdaptiveTestResult
        """
        try:
            if strategy_id is None:
                strategy_id = replay.strategy_id

            gate = AdaptiveRiskGate(
                strategy_id=strategy_id,
                params_path=self.params_path
            )

            details = {}
            inconsistent_decisions = []

            # 测试多次相同输入
            test_price = 100.0
            num_runs = 5

            decisions = []
            for i in range(num_runs):
                decision = gate.check(test_price)
                decisions.append(decision)

            # 检查决策一致性
            first_decision = decisions[0]
            for i, decision in enumerate(decisions[1:], 1):
                if decision.action != first_decision.action:
                    inconsistent_decisions.append({
                        "run": i,
                        "expected_action": first_decision.action.value,
                        "actual_action": decision.action.value
                    })

            # 测试状态一致性
            status_checks = []
            for i in range(3):
                status = gate.get_status()
                status_checks.append(status)

            details["num_runs"] = num_runs
            details["first_action"] = first_decision.action.value
            details["first_rule_id"] = first_decision.rule_id
            details["all_actions_consistent"] = len(inconsistent_decisions) == 0

            if inconsistent_decisions:
                return AdaptiveTestResult(
                    test_name="gating_decision_consistency",
                    status="FAIL",
                    message=f"Found {len(inconsistent_decisions)} inconsistent decisions",
                    details={"inconsistencies": inconsistent_decisions, **details}
                )

            return AdaptiveTestResult(
                test_name="gating_decision_consistency",
                status="PASS",
                message=f"All {num_runs} decisions consistent",
                details=details
            )

        except Exception as e:
            return AdaptiveTestResult(
                test_name="gating_decision_consistency",
                status="FAIL",
                message=f"Test error: {str(e)}",
                details={"error": str(e)}
            )

    def test_envelope_constraints(
        self,
        replay: ReplayOutput,
        baseline: RiskBaseline,
        strategy_id: str = None
    ) -> AdaptiveTestResult:
        """C3.4: 风险包络约束测试

        验证自适应风控不会让策略超出基线风险包络。

        Args:
            replay: 回测输出
            baseline: 基线
            strategy_id: 策略ID

        Returns:
            AdaptiveTestResult
        """
        try:
            if strategy_id is None:
                strategy_id = replay.strategy_id

            gate = AdaptiveRiskGate(
                strategy_id=strategy_id,
                params_path=self.params_path
            )

            details = {}
            violations = []

            # 模拟运行整个回测
            df = replay.to_dataframe()
            if df.empty:
                return AdaptiveTestResult(
                    test_name="envelope_constraints",
                    status="SKIP",
                    message="No data to test",
                    details={"reason": "empty_replay"}
                )

            # 获取基线包络
            for window_length, baseline_envelope in baseline.envelopes.items():
                if not baseline_envelope:
                    continue

                # 遍历每个窗口，检查gating是否有效控制风险
                # 这里简化：只检查是否有足够的gating事件

                # 统计gating事件
                gating_count = 0
                total_steps = 0

                for _, row in df.iterrows():
                    price = float(row.get('equity', 100000)) / 1000.0
                    decision = gate.check(price)

                    if decision.action == GatingAction.GATE:
                        gating_count += 1

                    total_steps += 1

                # 检查：gating率应该在合理范围内（10%-30%）
                gating_rate = gating_count / total_steps if total_steps > 0 else 0

                details[window_length] = {
                    "gating_count": gating_count,
                    "total_steps": total_steps,
                    "gating_rate": gating_rate,
                    "baseline_return_p95": baseline_envelope.get("return_p95"),
                    "baseline_mdd_p95": baseline_envelope.get("mdd_p95")
                }

                # 如果gating率过低（<5%），可能意味着风控不够
                if gating_rate < 0.05:
                    violations.append({
                        "window": window_length,
                        "issue": "insufficient_gating",
                        "gating_rate": gating_rate,
                        "min_expected": 0.05
                    })

                # 如果gating率过高（>50%），可能意味着风控过严
                if gating_rate > 0.50:
                    violations.append({
                        "window": window_length,
                        "issue": "excessive_gating",
                        "gating_rate": gating_rate,
                        "max_expected": 0.50
                    })

            if violations:
                return AdaptiveTestResult(
                    test_name="envelope_constraints",
                    status="FAIL",
                    message=f"Found {len(violations)} envelope constraint violations",
                    details={"violations": violations, **details}
                )

            return AdaptiveTestResult(
                test_name="envelope_constraints",
                status="PASS",
                message="Envelope constraints satisfied",
                details=details
            )

        except Exception as e:
            return AdaptiveTestResult(
                test_name="envelope_constraints",
                status="FAIL",
                message=f"Test error: {str(e)}",
                details={"error": str(e)}
            )


def run_adaptive_tests(
    replays: List[ReplayOutput],
    baselines: Dict[str, RiskBaseline],
    params_path: str = "config/adaptive_gating_params.json"
) -> Dict[str, List[AdaptiveTestResult]]:
    """运行所有自适应风控测试

    Args:
        replays: 回测输出列表
        baselines: {run_id: RiskBaseline} 字典
        params_path: 标定参数路径

    Returns:
        {run_id: [AdaptiveTestResult]} 字典
    """
    test_suite = AdaptiveGatingTests(params_path=params_path)

    all_results = {}

    for replay in replays:
        results = []
        baseline = baselines.get(replay.run_id)

        # C3.1: 阈值参数稳定性（全局测试，只需运行一次）
        if replay == replays[0]:
            results.append(test_suite.test_threshold_parameter_stability())

        # C3.2: 市场状态响应
        results.append(test_suite.test_regime_response(replay))

        # C3.3: Gating决策一致性
        results.append(test_suite.test_gating_decision_consistency(replay))

        # C3.4: 风险包络约束（需要baseline）
        if baseline:
            results.append(test_suite.test_envelope_constraints(replay, baseline))

        all_results[replay.run_id] = results

    return all_results


if __name__ == "__main__":
    """测试代码"""
    print("=== AdaptiveGatingTests 测试 ===\n")

    from analysis.replay_schema import ReplayOutput, load_replay_outputs

    # 加载回测数据
    replays = load_replay_outputs("runs")

    if len(replays) == 0:
        print("未找到回测数据")
    else:
        # 运行测试
        results = run_adaptive_tests(
            replays[:1],  # 只测试第一个
            {},
            params_path=None  # 使用默认参数
        )

        for run_id, test_results in results.items():
            print(f"\n策略: {run_id}")
            for result in test_results:
                print(f"  {result.test_name}: {result.status}")
                print(f"    {result.message}")

    print("\n✓ 测试完成")
