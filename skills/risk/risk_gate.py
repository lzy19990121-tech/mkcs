"""
Runtime Risk Gate for SPL-4a

Enforces SPL-3b rules at runtime to protect against worst-case scenarios.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

from core.context import RunContext
from skills.risk.runtime_metrics import RuntimeRiskCalculator, RuntimeRiskMetrics
from analysis.actionable_rules import RiskRuleset, RiskRule, RuleType, RuleScope


class GateAction(Enum):
    """Actions that can be taken by risk gate"""
    PAUSE_TRADING = "pause_trading"
    REDUCE_POSITION = "reduce_position"
    DISABLE_STRATEGY = "disable_strategy"
    NO_ACTION = "no_action"


@dataclass
class GateDecision:
    """Decision from risk gate"""
    action: GateAction
    reason: str
    triggered_rules: List[str]
    position_reduction_ratio: Optional[float] = None  # For REDUCE_POSITION
    recovery_condition: Optional[str] = None  # When to lift gate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action": self.action.value,
            "reason": self.reason,
            "triggered_rules": self.triggered_rules,
            "position_reduction_ratio": self.position_reduction_ratio,
            "recovery_condition": self.recovery_condition,
            "timestamp": datetime.now().isoformat()
        }


class RiskGate:
    """Runtime risk gate that enforces SPL-3b rules

    Checks risk metrics on each tick and takes protective action
    when rules are triggered.

    Rule Priority:
    1. GATING (highest) - Pause trading immediately
    2. POSITION_REDUCTION - Reduce position size
    3. DISABLE - Disable strategy entirely
    """

    def __init__(
        self,
        ruleset: RiskRuleset,
        window_length: str = "20d",
        initial_cash: float = 1000000.0
    ):
        """Initialize risk gate

        Args:
            ruleset: Risk ruleset from SPL-3b analysis
            window_length: Analysis window length
            initial_cash: Initial capital
        """
        self.ruleset = ruleset
        self.window_length = window_length
        self.metric_calculator = RuntimeRiskCalculator(
            window_length=window_length,
            initial_cash=initial_cash
        )

        # Active gate state
        self.active_gates: Dict[str, GateDecision] = {}
        self.is_paused: bool = False
        self.is_disabled: bool = False

        # Statistics
        self.total_checks: int = 0
        self.gate_triggers: int = 0

    def check(
        self,
        ctx: RunContext,
        positions: Dict,
        cash: float
    ) -> GateDecision:
        """Check all rules and return gate decision

        Args:
            ctx: Run context
            positions: Current positions
            cash: Current cash balance

        Returns:
            GateDecision with action to take
        """
        self.total_checks += 1

        # If strategy is disabled, always return disable
        if self.is_disabled:
            return GateDecision(
                action=GateAction.DISABLE_STRATEGY,
                reason="Strategy is disabled",
                triggered_rules=["DISABLED"]
            )

        # If paused, check if we can resume
        if self.is_paused:
            metrics = self.metric_calculator.calculate(ctx, positions, cash)
            if self._can_resume(metrics):
                self.is_paused = False
                # Clear pause gate
                self.active_gates.pop("pause", None)
                return GateDecision(
                    action=GateAction.NO_ACTION,
                    reason="Trading resumed - metrics recovered",
                    triggered_rules=[]
                )
            else:
                return GateDecision(
                    action=GateAction.PAUSE_TRADING,
                    reason="Trading paused - metrics not recovered",
                    triggered_rules=["PAUSED"]
                )

        # Calculate current metrics
        metrics = self.metric_calculator.calculate(ctx, positions, cash)

        # Check each rule in priority order
        sorted_rules = sorted(
            self.ruleset.rules,
            key=lambda r: self._rule_priority(r)
        )

        for rule in sorted_rules:
            if self._rule_triggered(rule, metrics, positions, cash):
                self.gate_triggers += 1

                # Create gate decision
                decision = self._create_gate_decision(rule, metrics)

                # Update state
                if decision.action == GateAction.PAUSE_TRADING:
                    self.is_paused = True
                    self.active_gates["pause"] = decision
                elif decision.action == GateAction.DISABLE_STRATEGY:
                    self.is_disabled = True
                    self.active_gates["disable"] = decision

                return decision

        # No rules triggered
        return GateDecision(
            action=GateAction.NO_ACTION,
            reason="All rules passed",
            triggered_rules=[]
        )

    def _rule_priority(self, rule: RiskRule) -> int:
        """Define rule priority (lower = higher priority)

        Args:
            rule: Risk rule

        Returns:
            Priority value
        """
        if rule.rule_type == RuleType.GATING:
            return 1  # Highest priority
        elif rule.rule_type == RuleType.POSITION_REDUCTION:
            return 2
        else:  # DISABLE
            return 3

    def _rule_triggered(
        self,
        rule: RiskRule,
        metrics: RuntimeRiskMetrics,
        positions: Dict,
        cash: float
    ) -> bool:
        """Check if a rule is triggered

        Args:
            rule: Risk rule to check
            metrics: Current risk metrics
            positions: Current positions
            cash: Current cash

        Returns:
            True if rule is triggered
        """
        # Get metric value
        metric_value = self._get_metric_value(
            rule.trigger_metric, metrics, positions, cash
        )

        if metric_value is None:
            return False

        # Check threshold
        threshold = rule.trigger_threshold
        operator = rule.trigger_operator

        if operator == "<":
            return metric_value < threshold
        elif operator == "<=":
            return metric_value <= threshold
        elif operator == ">":
            return metric_value > threshold
        elif operator == ">=":
            return metric_value >= threshold
        elif operator == "==":
            return metric_value == threshold
        else:
            return False

    def _get_metric_value(
        self,
        metric_name: str,
        metrics: RuntimeRiskMetrics,
        positions: Dict,
        cash: float
    ) -> Optional[float]:
        """Get current value of a metric

        Args:
            metric_name: Name of metric
            metrics: Current metrics
            positions: Current positions
            cash: Current cash

        Returns:
            Metric value or None if not found
        """
        # Map metric names to RuntimeRiskMetrics fields
        metric_map = {
            "stability_score": metrics.rolling_stability_score,
            "window_return": metrics.rolling_window_return,
            "max_drawdown": metrics.rolling_max_drawdown,
            "drawdown_duration": metrics.rolling_drawdown_duration,
            "return_volatility": metrics.rolling_return_volatility,
            "total_exposure": metrics.total_exposure
        }

        return metric_map.get(metric_name)

    def _create_gate_decision(
        self,
        rule: RiskRule,
        metrics: RuntimeRiskMetrics
    ) -> GateDecision:
        """Create gate decision from triggered rule

        Args:
            rule: Triggered rule
            metrics: Current metrics

        Returns:
            GateDecision
        """
        action_map = {
            RuleType.GATING: GateAction.PAUSE_TRADING,
            RuleType.POSITION_REDUCTION: GateAction.REDUCE_POSITION,
            RuleType.DISABLE: GateAction.DISABLE_STRATEGY
        }

        action = action_map.get(rule.rule_type, GateAction.NO_ACTION)

        # Create recovery condition
        recovery_condition = self._create_recovery_condition(rule, metrics)

        # For position reduction, default to 50% reduction
        reduction_ratio = 0.5 if action == GateAction.REDUCE_POSITION else None

        return GateDecision(
            action=action,
            reason=f"Rule triggered: {rule.rule_name}",
            triggered_rules=[rule.rule_id],
            position_reduction_ratio=reduction_ratio,
            recovery_condition=recovery_condition
        )

    def _create_recovery_condition(
        self,
        rule: RiskRule,
        metrics: RuntimeRiskMetrics
    ) -> str:
        """Create recovery condition description

        Args:
            rule: Triggered rule
            metrics: Current metrics

        Returns:
            Recovery condition description
        """
        if rule.trigger_metric == "stability_score":
            return f"Stability score recovers above {rule.trigger_threshold + 10:.0f}"
        elif rule.trigger_metric == "window_return":
            return f"Window return recovers above {rule.trigger_threshold * 2:.1%}"
        elif rule.trigger_metric == "max_drawdown":
            return f"Max drawdown falls below {rule.trigger_threshold * 0.5:.1%}"
        elif rule.trigger_metric == "drawdown_duration":
            return f"Drawdown duration falls below {rule.trigger_threshold * 0.5:.0f} days"
        else:
            return "Metrics improve significantly"

    def _can_resume(self, metrics: RuntimeRiskMetrics) -> bool:
        """Check if trading can resume after pause

        Args:
            metrics: Current metrics

        Returns:
            True if can resume
        """
        # Check all GATING rules to see if metrics have recovered
        for rule in self.ruleset.get_rules_by_type(RuleType.GATING):
            if self._rule_triggered(rule, metrics, {}, 0):
                return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get gate statistics

        Returns:
            Statistics dictionary
        """
        return {
            "total_checks": self.total_checks,
            "gate_triggers": self.gate_triggers,
            "trigger_rate": self.gate_triggers / self.total_checks if self.total_checks > 0 else 0,
            "is_paused": self.is_paused,
            "is_disabled": self.is_disabled,
            "active_gates": len(self.active_gates)
        }

    def reset(self):
        """Reset gate state"""
        self.active_gates.clear()
        self.is_paused = False
        self.is_disabled = False
        self.metric_calculator.reset()


if __name__ == "__main__":
    """Test risk gate"""
    print("=== RiskGate Test ===\n")

    from analysis.actionable_rules import RiskRule, RiskRuleset, RuleType, RuleScope
    from unittest.mock import Mock
    from datetime import datetime

    # Create test ruleset
    ruleset = RiskRuleset(strategy_id="test_strategy")

    ruleset.add_rule(RiskRule(
        rule_id="test_stability",
        rule_name="Low Stability Pause",
        rule_type=RuleType.GATING,
        scope=RuleScope.STRATEGY_LEVEL,
        trigger_metric="stability_score",
        trigger_threshold=50.0,
        trigger_operator="<",
        risk_type="Low stability",
        evidence_source="Test",
        evidence_value=30.0,
        applicable_strategy="test_strategy",
        applicable_window="all",
        description="Test rule"
    ))

    ruleset.add_rule(RiskRule(
        rule_id="test_drawdown",
        rule_name="High Drawdown Disable",
        rule_type=RuleType.DISABLE,
        scope=RuleScope.WINDOW_LEVEL,
        trigger_metric="max_drawdown",
        trigger_threshold=0.15,
        trigger_operator=">",
        risk_type="High drawdown",
        evidence_source="Test",
        evidence_value=0.20,
        applicable_strategy="test_strategy",
        applicable_window="20d",
        description="Test rule"
    ))

    # Create gate
    gate = RiskGate(ruleset, window_length="20d", initial_cash=1000000.0)

    # Mock context
    ctx = Mock()
    ctx.now = datetime.now()
    ctx.trading_date = ctx.now.date()

    positions = {}
    cash = 1000000.0

    # Simulate normal conditions
    print("Test 1: Normal conditions")
    decision = gate.check(ctx, positions, cash)
    print(f"  Action: {decision.action.value}")
    print(f"  Reason: {decision.reason}")
    assert decision.action == GateAction.NO_ACTION

    # Simulate declining equity (will trigger stability rule)
    print("\nTest 2: Declining equity")
    for i in range(25):
        cash -= 10000  # Decline by 1% each tick
        decision = gate.check(ctx, positions, cash)

    print(f"  Action: {decision.action.value}")
    print(f"  Reason: {decision.reason}")
    print(f"  Is Paused: {gate.is_paused}")

    stats = gate.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total Checks: {stats['total_checks']}")
    print(f"  Gate Triggers: {stats['gate_triggers']}")
    print(f"  Trigger Rate: {stats['trigger_rate']*100:.1f}%")

    print("\n✓ Test passed")
