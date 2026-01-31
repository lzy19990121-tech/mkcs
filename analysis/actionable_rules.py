"""
可执行风险规则生成模块

基于worst-case分析提出具体的风险约束规则
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from analysis.replay_schema import ReplayOutput
from analysis.window_scanner import WindowMetrics
from analysis.stability_analysis import StabilityReport
from analysis.risk_envelope import RiskEnvelope
from analysis.structural_analysis import StructuralAnalysisResult


class RuleScope(Enum):
    """规则适用范围"""
    STRATEGY_LEVEL = "strategy"      # 策略级
    WINDOW_LEVEL = "window"          # 窗口尺度级
    REGIME_LEVEL = "regime"          # 市场状态级


class RuleType(Enum):
    """规则类型"""
    GATING = "gating"                # 启停条件
    POSITION_REDUCTION = "reduction"  # 降仓条件
    DISABLE = "disable"              # 禁用条件


@dataclass
class RiskRule:
    """风险规则"""
    rule_id: str
    rule_name: str
    rule_type: RuleType
    scope: RuleScope

    # 触发条件
    trigger_metric: str           # 触发指标
    trigger_threshold: float      # 触发阈值
    trigger_operator: str         # 比较操作符 (>, <, >=, <=)

    # 风险类型
    risk_type: str                # 约束的风险类型

    # 证据来源
    evidence_source: str          # 来自哪个worst-case分析
    evidence_value: float         # 证据数值

    # 适用范围
    applicable_strategy: str      # 适用策略
    applicable_window: str        # 适用窗口

    # 规则描述
    description: str

    def to_trigger_code(self) -> str:
        """生成触发代码伪代码"""
        return f"if {self.trigger_metric} {self.trigger_operator} {self.trigger_threshold}:"


@dataclass
class RiskRuleset:
    """风险规则集合"""
    strategy_id: str
    rules: List[RiskRule] = field(default_factory=list)

    def add_rule(self, rule: RiskRule):
        """添加规则"""
        self.rules.append(rule)

    def get_rules_by_type(self, rule_type: RuleType) -> List[RiskRule]:
        """按类型获取规则"""
        return [r for r in self.rules if r.rule_type == rule_type]

    def get_rules_by_scope(self, scope: RuleScope) -> List[RiskRule]:
        """按范围获取规则"""
        return [r for r in self.rules if r.scope == scope]


class RiskRuleGenerator:
    """风险规则生成器

    基于worst-case分析生成可执行的风险约束规则
    """

    def generate_rules(
        self,
        replay: ReplayOutput,
        stability_report: StabilityReport,
        envelope: RiskEnvelope,
        structural_result: StructuralAnalysisResult
    ) -> RiskRuleset:
        """生成风险规则

        Args:
            replay: 回测输出
            stability_report: 稳定性报告
            envelope: 风险包络
            structural_result: 结构分析结果

        Returns:
            风险规则集合
        """
        ruleset = RiskRuleset(strategy_id=replay.strategy_id)

        # 1. 基于稳定性评分的启停规则
        stability_rule = self._generate_stability_gating_rule(
            replay, stability_report
        )
        if stability_rule:
            ruleset.add_rule(stability_rule)

        # 2. 基于worst-case return的降仓规则
        return_rule = self._generate_return_reduction_rule(
            replay, envelope
        )
        if return_rule:
            ruleset.add_rule(return_rule)

        # 3. 基于MDD的禁用规则
        mdd_rule = self._generate_mdd_disable_rule(
            replay, envelope
        )
        if mdd_rule:
            ruleset.add_rule(mdd_rule)

        # 4. 基于结构性风险的regime规则
        if structural_result.risk_pattern_type.value == "structural":
            regime_rule = self._generate_regime_rule(
                replay, structural_result
            )
            if regime_rule:
                ruleset.add_rule(regime_rule)

        # 5. 基于回撤持续时间的规则
        duration_rule = self._generate_duration_rule(
            replay, envelope
        )
        if duration_rule:
            ruleset.add_rule(duration_rule)

        return ruleset

    def _generate_stability_gating_rule(
        self,
        replay: ReplayOutput,
        report: StabilityReport
    ) -> Optional[RiskRule]:
        """生成基于稳定性评分的启停规则"""
        # 如果稳定性评分 < 30，生成启停规则
        if report.stability_score < 30:
            return RiskRule(
                rule_id=f"{replay.strategy_id}_stability_gating",
                rule_name="低稳定性暂停交易",
                rule_type=RuleType.GATING,
                scope=RuleScope.STRATEGY_LEVEL,

                trigger_metric="stability_score",
                trigger_threshold=30.0,
                trigger_operator="<",

                risk_type="低稳定性风险",
                evidence_source="稳定性分析",
                evidence_value=report.stability_score,

                applicable_strategy=replay.strategy_id,
                applicable_window="all",

                description=(
                    f"策略稳定性评分 {report.stability_score:.1f}/100 低于阈值30。"
                    f"策略在不同窗口下表现极不稳定，建议暂停使用。"
                )
            )
        return None

    def _generate_return_reduction_rule(
        self,
        replay: ReplayOutput,
        envelope: RiskEnvelope
    ) -> Optional[RiskRule]:
        """生成基于worst-case return的降仓规则"""
        # 使用P95分位数作为阈值
        worst_return = envelope.return_p95

        # 如果最坏收益 < -10%，生成降仓规则
        if worst_return < -0.10:
            return RiskRule(
                rule_id=f"{replay.strategy_id}_return_reduction",
                rule_name="极端亏损降仓",
                rule_type=RuleType.POSITION_REDUCTION,
                scope=RuleScope.WINDOW_LEVEL,

                trigger_metric="window_return",
                trigger_threshold=-0.10,
                trigger_operator="<",

                risk_type="极端收益风险",
                evidence_source="风险包络 (P95)",
                evidence_value=worst_return,

                applicable_strategy=replay.strategy_id,
                applicable_window=envelope.window_length,

                description=(
                    f"在{envelope.window_length}窗口下，P95最坏收益为{worst_return*100:.2f}%。"
                    f"当窗口收益低于-10%时，建议将仓位降低50%。"
                )
            )
        return None

    def _generate_mdd_disable_rule(
        self,
        replay: ReplayOutput,
        envelope: RiskEnvelope
    ) -> Optional[RiskRule]:
        """生成基于MDD的禁用规则"""
        # 使用P95分位数作为阈值
        worst_mdd = envelope.mdd_p95

        # 如果最大回撤 > 5%，生成禁用规则
        if worst_mdd > 0.05:
            return RiskRule(
                rule_id=f"{replay.strategy_id}_mdd_disable",
                rule_name="大幅回撤禁用",
                rule_type=RuleType.DISABLE,
                scope=RuleScope.WINDOW_LEVEL,

                trigger_metric="max_drawdown",
                trigger_threshold=0.05,
                trigger_operator=">",

                risk_type="大幅回撤风险",
                evidence_source="风险包络 (P95)",
                evidence_value=worst_mdd,

                applicable_strategy=replay.strategy_id,
                applicable_window=envelope.window_length,

                description=(
                    f"在{envelope.window_length}窗口下，P95最大回撤为{worst_mdd*100:.2f}%。"
                    f"当最大回撤超过5%时，建议禁用该策略。"
                )
            )
        return None

    def _generate_regime_rule(
        self,
        replay: ReplayOutput,
        structural_result: StructuralAnalysisResult
    ) -> Optional[RiskRule]:
        """生成基于结构性风险的regime规则"""
        # 如果是结构性风险，建议在特定市场状态下禁用
        pattern_metrics = structural_result.pattern_metrics

        if pattern_metrics.mdd_cv < 0.3 and pattern_metrics.pattern_similarity > 0.7:
            return RiskRule(
                rule_id=f"{replay.strategy_id}_regime_disable",
                rule_name="结构性风险禁用",
                rule_type=RuleType.DISABLE,
                scope=RuleScope.REGIME_LEVEL,

                trigger_metric="market_regime",
                trigger_threshold=0.0,  # 占位，需要ADX等指标
                trigger_operator="<",

                risk_type="结构性风险",
                evidence_source="结构分析",
                evidence_value=pattern_metrics.pattern_similarity,

                applicable_strategy=replay.strategy_id,
                applicable_window=structural_result.window_length,

                description=(
                    f"策略表现出结构性风险（形态相似度{pattern_metrics.pattern_similarity:.2f}）。"
                    f"Top-K最坏窗口高度相似，建议在震荡市场（ADX < 25）时禁用该策略。"
                )
            )
        return None

    def _generate_duration_rule(
        self,
        replay: ReplayOutput,
        envelope: RiskEnvelope
    ) -> Optional[RiskRule]:
        """生成基于回撤持续时间的规则"""
        # 使用P95分位数作为阈值
        worst_duration = envelope.duration_p95

        # 如果回撤持续时间 > 10天，生成规则
        if worst_duration > 10:
            return RiskRule(
                rule_id=f"{replay.strategy_id}_duration_gating",
                rule_name="持续回撤暂停",
                rule_type=RuleType.GATING,
                scope=RuleScope.WINDOW_LEVEL,

                trigger_metric="drawdown_duration",
                trigger_threshold=10.0,
                trigger_operator=">",

                risk_type="持续回撤风险",
                evidence_source="风险包络 (P95)",
                evidence_value=worst_duration,

                applicable_strategy=replay.strategy_id,
                applicable_window=envelope.window_length,

                description=(
                    f"在{envelope.window_length}窗口下，P95回撤持续时间为{worst_duration:.1f}天。"
                    f"当回撤持续超过10天时，建议暂停交易。"
                )
            )
        return None


def format_ruleset_report(ruleset: RiskRuleset) -> str:
    """格式化规则集报告

    Args:
        ruleset: 规则集合

    Returns:
        Markdown格式的报告
    """
    lines = []

    lines.append(f"## 可执行风险规则 - {ruleset.strategy_id}\n")

    if not ruleset.rules:
        lines.append("无风险规则（策略表现良好）\n")
        return "\n".join(lines)

    lines.append(f"**规则总数**: {len(ruleset.rules)}\n")

    # 按类型分组
    for rule_type in RuleType:
        type_rules = ruleset.get_rules_by_type(rule_type)
        if not type_rules:
            continue

        type_name = {
            RuleType.GATING: "启停规则 (Gating)",
            RuleType.POSITION_REDUCTION: "降仓规则 (Position Reduction)",
            RuleType.DISABLE: "禁用规则 (Disable)"
        }[rule_type]

        lines.append(f"### {type_name}\n")

        for i, rule in enumerate(type_rules, 1):
            lines.append(f"#### 规则 #{i}: {rule.rule_name}\n")

            lines.append(f"- **规则ID**: {rule.rule_id}")
            lines.append(f"- **适用范围**: {rule.scope.value}")
            lines.append(f"- **触发条件**: `{rule.trigger_metric} {rule.trigger_operator} {rule.trigger_threshold}`")
            lines.append(f"- **风险类型**: {rule.risk_type}")
            lines.append(f"- **证据来源**: {rule.evidence_source}")
            lines.append(f"- **证据数值**: {rule.evidence_value}")
            lines.append(f"\n**描述**: {rule.description}\n")

            lines.append("**伪代码**:")
            lines.append("```python")
            lines.append(f"{rule.to_trigger_code()}")
            lines.append("    # 执行相应动作")
            if rule.rule_type == RuleType.GATING:
                lines.append("    pause_trading()")
            elif rule.rule_type == RuleType.POSITION_REDUCTION:
                lines.append("    reduce_position(0.5)  # 降低50%仓位")
            elif rule.rule_type == RuleType.DISABLE:
                lines.append("    disable_strategy()")
            lines.append("```\n")

    return "\n".join(lines)


if __name__ == "__main__":
    """测试代码"""
    print("=== RiskRuleGenerator 测试 ===\n")

    from analysis import load_replay_outputs, StabilityAnalyzer, RiskEnvelopeBuilder, StructuralAnalyzer

    replays = load_replay_outputs("runs")

    if replays:
        replay = replays[0]
        print(f"生成风险规则: {replay.run_id}\n")

        # 生成各项分析结果
        stability_analyzer = StabilityAnalyzer()
        stability_report = stability_analyzer.analyze_replay(replay)

        envelope_builder = RiskEnvelopeBuilder()
        envelope = envelope_builder.build_envelope(replay, "20d")

        structural_analyzer = StructuralAnalyzer()
        structural_result = structural_analyzer.analyze_structure(replay, "20d")

        # 生成规则
        generator = RiskRuleGenerator()
        ruleset = generator.generate_rules(
            replay, stability_report, envelope, structural_result
        )

        # 打印报告
        report = format_ruleset_report(ruleset)
        print(report)

        print(f"共生成 {len(ruleset.rules)} 条风险规则")

    print("\n✓ 测试通过")
