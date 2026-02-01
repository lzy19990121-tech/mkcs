"""
测试 SPL-4a Runtime Gating 功能

验证风控规则在实际运行中��否有效触发
"""

import json
from datetime import datetime, date
from decimal import Decimal
from typing import Dict, Any

from skills.risk.risk_gate import RiskGate, GateAction
from analysis.actionable_rules import RiskRuleset, RiskRule, RuleType, RuleScope
from core.context import RunContext


def load_rules_from_json(json_path: str, window_length: str = "20d") -> RiskRuleset:
    """从深度分析 JSON 文件加载规则

    Args:
        json_path: JSON 文件路径
        window_length: 窗口长度

    Returns:
        RiskRuleset 对象
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    strategy_id = data.get("strategy_id", "unknown")
    ruleset = RiskRuleset(strategy_id=strategy_id)

    # 从指定窗口加载规则
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
            trigger_operator="<=",  # 默认操作符
            risk_type="worst_case",
            evidence_source=f"deep_analysis_v3b_{window_length}",
            evidence_value=rule_data["trigger_threshold"],
            applicable_strategy=strategy_id,
            applicable_window=window_length,
            description=rule_data["description"]
        )
        ruleset.add_rule(rule)

    return ruleset


def create_mock_context(
    current_date: date = None
) -> RunContext:
    """创建模拟的 RunContext

    Args:
        current_date: 当前日期

    Returns:
        RunContext 对象
    """
    if current_date is None:
        current_date = date(2024, 6, 30)

    # 创建 RunContext
    ctx = RunContext(
        now=datetime.combine(current_date, datetime.min.time()),
        trading_date=current_date,
        mode="replay",
        bar_interval="1d"
    )

    return ctx


def test_gating_scenario(
    gate: RiskGate,
    scenario_name: str,
    window_return: float,
    stability_score: float,
    expected_action: GateAction
) -> bool:
    """测试单个风控场景

    Args:
        gate: RiskGate 实例
        scenario_name: 场景名称
        window_return: 窗口收益
        stability_score: 稳定性评分
        expected_action: 期望的决策

    Returns:
        是否通过测试
    """
    print(f"\n  场景: {scenario_name}")
    print(f"    窗口收益: {window_return*100:.2f}%")
    print(f"    稳定性评分: {stability_score:.1f}")

    ctx = create_mock_context(
        window_return=window_return,
        stability_score=stability_score
    )

    decision = gate.check(ctx, {}, 100000.0)

    passed = decision.action == expected_action
    status = "✅ PASS" if passed else "❌ FAIL"

    print(f"    期望动作: {expected_action.value}")
    print(f"    实际动作: {decision.action.value}")
    print(f"    触发规则: {decision.triggered_rules}")
    print(f"    原因: {decision.reason}")
    print(f"    {status}")

    return passed


def main():
    """主测试流程"""
    print("=" * 70)
    print("SPL-4a Runtime Gating 测试")
    print("=" * 70)

    # 1. 加载规则
    print("\n1. 加载深度分析规则...")
    ruleset = load_rules_from_json(
        "runs/deep_analysis_v3b/exp_1677b52a_deep_analysis_v3b.json",
        window_length="20d"
    )

    print(f"   已加载 {len(ruleset.rules)} 条规则:")
    for rule in ruleset.rules:
        print(f"     - {rule.rule_name}: {rule.trigger_metric} {rule.trigger_operator} {rule.trigger_threshold}")

    # 2. 创建 RiskGate
    print("\n2. 创建 RiskGate...")
    gate = RiskGate(
        ruleset=ruleset,
        window_length="20d",
        initial_cash=100000.0
    )
    print("   ✓ RiskGate 创建成功")

    # 3. 测试场景
    print("\n3. 测试风控场景...")

    test_cases = [
        {
            "name": "正常情况（无触发）",
            "window_return": 0.05,
            "stability_score": 80.0,
            "expected": GateAction.NO_ACTION
        },
        {
            "name": "低稳定性（触发暂停）",
            "window_return": 0.05,
            "stability_score": 20.0,
            "expected": GateAction.PAUSE_TRADING
        },
        {
            "name": "极端亏损（触发降仓）",
            "window_return": -0.15,
            "stability_score": 80.0,
            "expected": GateAction.REDUCE_POSITION
        },
        {
            "name": "低稳定性 + 极端亏损（触发暂停，优先级更高）",
            "window_return": -0.15,
            "stability_score": 20.0,
            "expected": GateAction.PAUSE_TRADING
        }
    ]

    passed = 0
    failed = 0

    for test_case in test_cases:
        if test_gating_scenario(
            gate,
            test_case["name"],
            test_case["window_return"],
            test_case["stability_score"],
            test_case["expected"]
        ):
            passed += 1
        else:
            failed += 1

    # 4. 统计信息
    print("\n" + "=" * 70)
    print("测试统计")
    print("=" * 70)
    stats = gate.get_statistics()
    print(f"总检查次数: {stats['total_checks']}")
    print(f"风控触发次数: {stats['gate_triggers']}")
    print(f"触发率: {stats['trigger_rate']*100:.2f}%")
    print(f"\n测试通过: {passed}/{passed+failed}")

    if failed == 0:
        print("\n✅ 所有测试通过！Runtime Gating 功能正常。")
    else:
        print(f"\n❌ {failed} 个测试失败，请检查配置。")

    print("=" * 70)


# 添加 get_statistics 方法到 RiskGate（如果尚未实现）
if not hasattr(RiskGate, 'get_statistics'):
    def get_statistics(self) -> Dict[str, Any]:
        """获取风控统计信息"""
        return {
            "total_checks": self.total_checks,
            "gate_triggers": self.gate_triggers,
            "trigger_rate": self.gate_triggers / self.total_checks if self.total_checks > 0 else 0.0,
            "is_paused": self.is_paused,
            "is_disabled": self.is_disabled,
            "active_gates": len(self.active_gates)
        }

    RiskGate.get_statistics = get_statistics


if __name__ == "__main__":
    main()
