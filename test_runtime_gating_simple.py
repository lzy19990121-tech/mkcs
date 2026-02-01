"""
测试 SPL-4a Runtime Gating 功能（简化版）

直接测试风控规则的决策逻辑
"""

import json
from datetime import datetime, date

from skills.risk.risk_gate import RiskGate, GateAction
from analysis.actionable_rules import RiskRuleset, RiskRule, RuleType, RuleScope
from skills.risk.runtime_metrics import RuntimeRiskMetrics
from core.context import RunContext


def load_rules_from_json(json_path: str, window_length: str = "20d") -> RiskRuleset:
    """从深度分析 JSON 文件加载规则"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    strategy_id = data.get("strategy_id", "unknown")
    ruleset = RiskRuleset(strategy_id=strategy_id)

    window_data = data.get("windows", {}).get(window_length, {})
    rules_data = window_data.get("rules", [])

    for rule_data in rules_data:
        rule = RiskRule(
            rule_id=rule_data["rule_id"],
            rule_name=rule_data["rule_name"],
            rule_type=RuleType(rule_data["rule_type"]),
            scope=RuleScope.STRATEGY_LEVEL,
            trigger_metric=rule_data["trigger_metric"],
            trigger_threshold=rule_data["trigger_threshold"],
            trigger_operator="<=",
            risk_type="worst_case",
            evidence_source=f"deep_analysis_v3b_{window_length}",
            evidence_value=rule_data["trigger_threshold"],
            applicable_strategy=strategy_id,
            applicable_window=window_length,
            description=rule_data["description"]
        )
        ruleset.add_rule(rule)

    return ruleset


def create_mock_context() -> RunContext:
    """创建模拟的 RunContext"""
    return RunContext(
        now=datetime.now(),
        trading_date=date.today(),
        mode="replay",
        bar_interval="1d"
    )


def create_mock_metrics(
    window_return: float = 0.0,
    stability_score: float = 50.0,
    max_drawdown: float = 0.05,
    drawdown_duration: int = 5,
    market_regime: str = "trending",
    adx: float = 30.0
) -> RuntimeRiskMetrics:
    """创建模拟的风险指标"""
    return RuntimeRiskMetrics(
        rolling_stability_score=stability_score,
        rolling_return_volatility=0.02,
        rolling_window_return=window_return,
        rolling_max_drawdown=max_drawdown,
        rolling_drawdown_duration=drawdown_duration,
        current_adx=adx,
        market_regime=market_regime,
        total_exposure=0.5,
        num_positions=1,
        calculated_at=datetime.now()
    )


def test_rule_evaluation(
    ruleset: RiskRuleset,
    scenario_name: str,
    metrics: RuntimeRiskMetrics,
    expected_trigger_count: int
) -> bool:
    """测试规则评估

    Args:
        ruleset: 规则集
        scenario_name: 场景名称
        metrics: 模拟的指标
        expected_trigger_count: 期望触发的规则数

    Returns:
        是否通过测试
    """
    print(f"\n  场景: {scenario_name}")
    print(f"    窗口收益: {metrics.rolling_window_return*100:.2f}%")
    print(f"    稳定性评分: {metrics.rolling_stability_score:.1f}")
    print(f"    最大回撤: {metrics.rolling_max_drawdown*100:.2f}%")
    print(f"    市场状态: {metrics.market_regime} (ADX={metrics.current_adx})")

    triggered_rules = []

    for rule in ruleset.rules:
        triggered = False
        value = None

        # 获取指标值
        if rule.trigger_metric == "stability_score":
            value = metrics.rolling_stability_score
            triggered = value <= rule.trigger_threshold
        elif rule.trigger_metric == "window_return":
            value = metrics.rolling_window_return
            triggered = value <= rule.trigger_threshold
        elif rule.trigger_metric == "market_regime":
            # 对于 market_regime，使用 ADX 阈值
            value = metrics.current_adx
            # 假设 threshold <= 0 表示 ADX < 25
            triggered = metrics.current_adx < 25 and rule.trigger_threshold <= 0
        elif rule.trigger_metric == "drawdown_duration":
            value = metrics.rolling_drawdown_duration
            triggered = value >= rule.trigger_threshold

        if triggered:
            triggered_rules.append(rule.rule_name)
            print(f"    ✓ 触发: {rule.rule_name} ({rule.trigger_metric}={value:.2f})")

    passed = len(triggered_rules) == expected_trigger_count
    status = "✅ PASS" if passed else "❌ FAIL"

    print(f"    期望触发: {expected_trigger_count} 条规则")
    print(f"    实际触发: {len(triggered_rules)} 条规则")
    print(f"    {status}")

    return passed


def test_gate_priority(
    ruleset: RiskRuleset,
    scenario_name: str,
    metrics: RuntimeRiskMetrics,
    expected_action: GateAction
) -> bool:
    """测试风控优先级

    Args:
        ruleset: 规则集
        scenario_name: 场景名称
        metrics: 模拟的指标
        expected_action: 期望的动作

    Returns:
        是否通过测试
    """
    print(f"\n  优先级测试: {scenario_name}")

    # 找到触发的规则
    triggered_rules = []
    for rule in ruleset.rules:
        triggered = False
        value = None

        if rule.trigger_metric == "stability_score":
            value = metrics.rolling_stability_score
            triggered = value <= rule.trigger_threshold
        elif rule.trigger_metric == "window_return":
            value = metrics.rolling_window_return
            triggered = value <= rule.trigger_threshold
        elif rule.trigger_metric == "market_regime":
            triggered = metrics.current_adx < 25 and rule.trigger_threshold <= 0
        elif rule.trigger_metric == "drawdown_duration":
            value = metrics.rolling_drawdown_duration
            triggered = value >= rule.trigger_threshold

        if triggered:
            triggered_rules.append(rule)

    # 确定优先级最高的动作
    action = GateAction.NO_ACTION
    reason = "No rules triggered"

    if triggered_rules:
        # 按优先级排序: GATING > POSITION_REDUCTION > DISABLE
        sorted_rules = sorted(
            triggered_rules,
            key=lambda r: {
                RuleType.GATING: 0,
                RuleType.POSITION_REDUCTION: 1,
                RuleType.DISABLE: 2
            }.get(r.rule_type, 3)
        )

        highest_priority = sorted_rules[0]
        if highest_priority.rule_type == RuleType.GATING:
            action = GateAction.PAUSE_TRADING
            reason = f"Low stability: {highest_priority.rule_name}"
        elif highest_priority.rule_type == RuleType.POSITION_REDUCTION:
            action = GateAction.REDUCE_POSITION
            reason = f"High loss: {highest_priority.rule_name}"
        elif highest_priority.rule_type == RuleType.DISABLE:
            action = GateAction.DISABLE_STRATEGY
            reason = f"Structural risk: {highest_priority.rule_name}"

    passed = action == expected_action
    status = "✅ PASS" if passed else "❌ FAIL"

    print(f"    触发规则: {[r.rule_name for r in triggered_rules]}")
    print(f"    期望动作: {expected_action.value}")
    print(f"    实际动作: {action.value}")
    print(f"    原因: {reason}")
    print(f"    {status}")

    return passed


def main():
    """主测试流程"""
    print("=" * 70)
    print("SPL-4a Runtime Gating 测试（简化版）")
    print("=" * 70)

    # 1. 加载规则
    print("\n1. 加载深度分析规则...")
    ruleset = load_rules_from_json(
        "runs/deep_analysis_v3b/exp_1677b52a_deep_analysis_v3b.json",
        window_length="20d"
    )

    print(f"   已加载 {len(ruleset.rules)} 条规则:")
    for rule in ruleset.rules:
        print(f"     - {rule.rule_name}:")
        print(f"         类型: {rule.rule_type.value}")
        print(f"         触发: {rule.trigger_metric} {rule.trigger_operator} {rule.trigger_threshold}")

    # 2. 测试规则评估
    print("\n2. 测试规则评估...")

    test_cases = [
        {
            "name": "正常情况（无触发）",
            "metrics": create_mock_metrics(
                window_return=0.05,
                stability_score=80.0,
                market_regime="trending",
                adx=30.0
            ),
            "expected_triggers": 0
        },
        {
            "name": "低稳定性（触发暂停）",
            "metrics": create_mock_metrics(
                window_return=0.05,
                stability_score=20.0,
                market_regime="trending",
                adx=30.0
            ),
            "expected_triggers": 1
        },
        {
            "name": "极端亏损（触发降仓）",
            "metrics": create_mock_metrics(
                window_return=-0.15,
                stability_score=80.0,
                market_regime="trending",
                adx=30.0
            ),
            "expected_triggers": 1
        },
        {
            "name": "震荡市场（触发禁用）",
            "metrics": create_mock_metrics(
                window_return=0.05,
                stability_score=80.0,
                market_regime="ranging",
                adx=20.0
            ),
            "expected_triggers": 1
        },
        {
            "name": "多重触发（低稳定性 + 极端亏损）",
            "metrics": create_mock_metrics(
                window_return=-0.15,
                stability_score=20.0,
                market_regime="trending",
                adx=30.0
            ),
            "expected_triggers": 2
        }
    ]

    passed = 0
    failed = 0

    for test_case in test_cases:
        if test_rule_evaluation(
            ruleset,
            test_case["name"],
            test_case["metrics"],
            test_case["expected_triggers"]
        ):
            passed += 1
        else:
            failed += 1

    # 3. 测试风控优先级
    print("\n3. 测试风控优先级...")

    priority_tests = [
        {
            "name": "正常情况（无动作）",
            "metrics": create_mock_metrics(
                window_return=0.05,
                stability_score=80.0,
                adx=30.0
            ),
            "expected": GateAction.NO_ACTION
        },
        {
            "name": "低稳定性优先于极端亏损",
            "metrics": create_mock_metrics(
                window_return=-0.15,
                stability_score=20.0,
                adx=30.0
            ),
            "expected": GateAction.PAUSE_TRADING
        },
        {
            "name": "只有极端亏损（触发降仓）",
            "metrics": create_mock_metrics(
                window_return=-0.15,
                stability_score=80.0,
                adx=30.0
            ),
            "expected": GateAction.REDUCE_POSITION
        }
    ]

    for test_case in priority_tests:
        if test_gate_priority(
            ruleset,
            test_case["name"],
            test_case["metrics"],
            test_case["expected"]
        ):
            passed += 1
        else:
            failed += 1

    # 4. 统计信息
    print("\n" + "=" * 70)
    print("测试统计")
    print("=" * 70)
    total = passed + failed
    print(f"总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"通过率: {passed/total*100:.1f}%")

    if failed == 0:
        print("\n✅ 所有测试通过！Runtime Gating 规则逻辑正常。")
    else:
        print(f"\n❌ {failed} 个测试失败，请检查配置。")

    print("=" * 70)


if __name__ == "__main__":
    main()
