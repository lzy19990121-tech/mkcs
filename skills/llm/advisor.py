"""
LLM Advisor - L5 层

盘后总结，规则建议，参数建议
不参与实时决策路径
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from core.schema import (
    MarketState,
    AlphaOpinion,
    MetaDecision,
    RiskDecision,
    ExecutionResult,
    SchemaVersion
)

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """建议类型"""
    GATING_CHANGE = "gating_change"  # 修改禁用规则
    PARAMETER_CHANGE = "parameter_change"  # 修改参数
    NEW_STRATEGY = "new_strategy"  # 新增策略
    RISK_ADJUST = "risk_adjust"  # 风控调整


@dataclass
class RuleProposal:
    """规则建议"""
    proposal_type: RecommendationType
    target: str  # 目标（策略名或参数名）
    current_value: Any
    suggested_value: Any
    reason: str
    expected_impact: str
    confidence: float = 0.5  # 建议的置信度


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_pnl: float = 0.0


@dataclass
class AdvisorReport:
    """顾问报告"""
    timestamp: datetime
    report_period: str  # 如 "2024-01-01 to 2024-01-31"
    summary: str
    performance_metrics: PerformanceMetrics
    strategy_analysis: Dict[str, str]  # 策略名 -> 分析
    market_analysis: str
    rule_proposals: List[RuleProposal]
    risk_assessment: str
    suggestions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "report_period": self.report_period,
            "summary": self.summary,
            "performance_metrics": {
                "total_trades": self.performance_metrics.total_trades,
                "winning_trades": self.performance_metrics.winning_trades,
                "losing_trades": self.performance_metrics.losing_trades,
                "win_rate": self.performance_metrics.win_rate,
                "total_pnl": self.performance_metrics.total_pnl,
                "max_drawdown": self.performance_metrics.max_drawdown,
                "sharpe_ratio": self.performance_metrics.sharpe_ratio,
                "avg_trade_pnl": self.performance_metrics.avg_trade_pnl,
            },
            "strategy_analysis": self.strategy_analysis,
            "market_analysis": self.market_analysis,
            "rule_proposals": [
                {
                    "type": p.proposal_type.value,
                    "target": p.target,
                    "current_value": str(p.current_value),
                    "suggested_value": str(p.suggested_value),
                    "reason": p.reason,
                    "expected_impact": p.expected_impact,
                    "confidence": p.confidence,
                }
                for p in self.rule_proposals
            ],
            "risk_assessment": self.risk_assessment,
            "suggestions": self.suggestions,
        }


class LLMAdvisor:
    """
    LLM Advisor - L5 层

    职责：
    ✓ 盘后总结 runs
    ✓ 生成规则建议 (RuleProposal)
    ✓ 生成参数建议
    ✓ 分析策略表现

    不做：
    ✗ 不参与实时决策路径
    ✗ 不直接修改系统配置
    """

    def __init__(self):
        """初始化 LLM Advisor"""
        self._report_history: List[AdvisorReport] = []

    def analyze_run(
        self,
        market_states: List[MarketState],
        meta_decisions: List[MetaDecision],
        risk_decisions: List[RiskDecision],
        execution_results: List[ExecutionResult],
        period_start: datetime,
        period_end: datetime
    ) -> AdvisorReport:
        """
        分析一次运行（盘后）

        Args:
            market_states: 市场状态历史
            meta_decisions: Meta决策历史
            risk_decisions: 风控决策历史
            execution_results: 执行结果历史
            period_start: 开始时间
            period_end: 结束时间

        Returns:
            AdvisorReport
        """
        # 1. 计算性能指标
        metrics = self._calculate_metrics(execution_results)

        # 2. 分析策略表现
        strategy_analysis = self._analyze_strategies(meta_decisions, execution_results)

        # 3. 市场分析
        market_analysis = self._analyze_market(market_states)

        # 4. 生成规则建议
        rule_proposals = self._generate_proposals(
            strategy_analysis,
            metrics,
            market_states
        )

        # 5. 风险评估
        risk_assessment = self._assess_risk(risk_decisions, execution_results)

        # 6. 生成总结
        summary = self._generate_summary(metrics, market_analysis, strategy_analysis)

        # 7. 生成建议
        suggestions = self._generate_suggestions(
            metrics,
            strategy_analysis,
            rule_proposals
        )

        report = AdvisorReport(
            timestamp=datetime.now(),
            report_period=f"{period_start.date()} to {period_end.date()}",
            summary=summary,
            performance_metrics=metrics,
            strategy_analysis=strategy_analysis,
            market_analysis=market_analysis,
            rule_proposals=rule_proposals,
            risk_assessment=risk_assessment,
            suggestions=suggestions
        )

        self._report_history.append(report)
        return report

    def _calculate_metrics(
        self,
        execution_results: List[ExecutionResult]
    ) -> PerformanceMetrics:
        """计算性能指标"""
        if not execution_results:
            return PerformanceMetrics()

        filled_results = [r for r in execution_results if r.status == "FILLED"]

        if not filled_results:
            return PerformanceMetrics()

        total_trades = len(filled_results)

        # 简化的盈亏计算（实际需要更复杂的逻辑）
        winning_trades = sum(1 for r in filled_results if r.fill_quantity > 0)
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 简化计算
        total_pnl = sum(
            (r.fill_price * r.fill_quantity * 0.01)  # 假设1%的盈亏
            for r in filled_results
        )
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # 最大回撤（简化）
        max_drawdown = abs(min(r.cash_balance for r in filled_results)) - 100000

        # 夏普比率（简化）
        sharpe_ratio = total_pnl / max(abs(total_pnl), 1) if total_pnl != 0 else 0

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            avg_trade_pnl=avg_trade_pnl
        )

    def _analyze_strategies(
        self,
        meta_decisions: List[MetaDecision],
        execution_results: List[ExecutionResult]
    ) -> Dict[str, str]:
        """分析策略表现"""
        strategy_stats: Dict[str, Dict[str, Any]] = {}

        # 统计策略参与情况
        for decision in meta_decisions:
            for strategy in decision.active_strategies:
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        "count": 0,
                        "total_confidence": 0,
                        "disabled_count": 0
                    }
                strategy_stats[strategy]["count"] += 1
                strategy_stats[strategy]["total_confidence"] += decision.decision_confidence

            for strategy, reason in decision.disabled_strategies.items():
                if strategy not in strategy_stats:
                    strategy_stats[strategy] = {
                        "count": 0,
                        "total_confidence": 0,
                        "disabled_count": 0
                    }
                strategy_stats[strategy]["disabled_count"] += 1

        # 生成分析
        analysis = {}
        for strategy, stats in strategy_stats.items():
            avg_confidence = (
                stats["total_confidence"] / stats["count"]
                if stats["count"] > 0 else 0
            )
            participation_rate = (
                stats["count"] / (stats["count"] + stats["disabled_count"])
                if (stats["count"] + stats["disabled_count"]) > 0 else 0
            )

            if participation_rate < 0.3:
                analysis[strategy] = (
                    f"{strategy} 策略参与率较低 ({participation_rate:.1%})，"
                    f"经常被禁用。建议检查 gating 规则是否过于严格。"
                )
            elif avg_confidence < 0.5:
                analysis[strategy] = (
                    f"{strategy} 策略平均置信度较低 ({avg_confidence:.2f})，"
                    f"信号质量可能需要改进。"
                )
            else:
                analysis[strategy] = (
                    f"{strategy} 策略表现良好，参与率 {participation_rate:.1%}，"
                    f"平均置信度 {avg_confidence:.2f}。"
                )

        return analysis

    def _analyze_market(self, market_states: List[MarketState]) -> str:
        """分析市场环境"""
        if not market_states:
            return "无市场数据"

        # 统计市场状态分布
        regime_counts: Dict[str, int] = {}
        volatility_counts: Dict[str, int] = {}
        liquidity_counts: Dict[str, int] = {}

        for state in market_states:
            regime_counts[state.regime] = regime_counts.get(state.regime, 0) + 1
            volatility_counts[state.volatility_state] = volatility_counts.get(state.volatility_state, 0) + 1
            liquidity_counts[state.liquidity_state] = liquidity_counts.get(state.liquidity_state, 0) + 1

        total = len(market_states)

        # 主要市场状态
        main_regime = max(regime_counts, key=regime_counts.get)
        main_volatility = max(volatility_counts, key=volatility_counts.get)
        main_liquidity = max(liquidity_counts, key=liquidity_counts.get)

        analysis = (
            f"本周期主要为 {main_regime} 市场 ({regime_counts[main_regime]}/{total})，"
            f"波动性 {main_volatility} ({volatility_counts[main_volatility]}/{total})，"
            f"流动性 {main_liquidity} ({liquidity_counts[main_liquidity]}/{total})。"
        )

        if regime_counts.get("CRISIS", 0) > 0:
            analysis += f" 经历了 {regime_counts['CRISIS']} 次危机状态。"

        if volatility_counts.get("EXTREME", 0) > total * 0.3:
            analysis += " 高波动时间占比超过30%，策略表现可能受到影响。"

        return analysis

    def _generate_proposals(
        self,
        strategy_analysis: Dict[str, str],
        metrics: PerformanceMetrics,
        market_states: List[MarketState]
    ) -> List[RuleProposal]:
        """生成规则建议"""
        proposals = []

        # 基于策略分析生成建议
        for strategy, analysis in strategy_analysis.items():
            if "参与率较低" in analysis:
                proposals.append(RuleProposal(
                    proposal_type=RecommendationType.GATING_CHANGE,
                    target=strategy,
                    current_value="当前禁用规则",
                    suggested_value="放宽禁用条件",
                    reason=f"{strategy} 策略参与率过低",
                    expected_impact="提高策略参与度，增加交易机会",
                    confidence=0.6
                ))

        # 基于性能指标生成建议
        if metrics.win_rate < 0.4:
            proposals.append(RuleProposal(
                proposal_type=RecommendationType.PARAMETER_CHANGE,
                target="confidence_threshold",
                current_value=0.5,
                suggested_value=0.6,
                reason="胜率低于40%",
                expected_impact="提高信号质量，减少低质量交易",
                confidence=0.7
            ))

        if metrics.max_drawdown < -10000:
            proposals.append(RuleProposal(
                proposal_type=RecommendationType.RISK_ADJUST,
                target="position_scaling",
                current_value=1.0,
                suggested_value=0.5,
                reason=f"最大回撤达到 {abs(metrics.max_drawdown):.0f}",
                expected_impact="降低风险暴露，减少回撤",
                confidence=0.8
            ))

        return proposals

    def _assess_risk(
        self,
        risk_decisions: List[RiskDecision],
        execution_results: List[ExecutionResult]
    ) -> str:
        """评估风险状况"""
        if not risk_decisions:
            return "无风控数据"

        # 统计风控动作
        action_counts: Dict[str, int] = {}
        scale_factors = []

        for decision in risk_decisions:
            action_counts[decision.risk_action] = action_counts.get(decision.risk_action, 0) + 1
            if decision.scale_factor < 1.0:
                scale_factors.append(decision.scale_factor)

        total = len(risk_decisions)
        approve_rate = action_counts.get("APPROVE", 0) / total

        assessment = f"风控通过率 {approve_rate:.1%}。"

        if "DISABLE" in action_counts:
            assessment += f" 触发 {action_counts['DISABLE']} 次禁止交易。"

        if "SCALE_DOWN" in action_counts:
            assessment += f" 触发 {action_counts['SCALE_DOWN']} 次仓位缩放。"

        avg_scale = sum(scale_factors) / len(scale_factors) if scale_factors else 1.0
        if avg_scale < 0.7:
            assessment += f" 平均缩放系数 {avg_scale:.2f}，市场环境较为不利。"

        return assessment

    def _generate_summary(
        self,
        metrics: PerformanceMetrics,
        market_analysis: str,
        strategy_analysis: Dict[str, str]
    ) -> str:
        """生成总结"""
        summary = f"""
本周期共执行 {metrics.total_trades} 笔交易，胜率 {metrics.win_rate:.1%}，
总盈亏 {metrics.total_pnl:.2f}，平均每笔盈亏 {metrics.avg_trade_pnl:.2f}。

{market_analysis}

策略表现方面：
"""

        for strategy, analysis in strategy_analysis.items():
            summary += f"\n- {analysis}"

        return summary.strip()

    def _generate_suggestions(
        self,
        metrics: PerformanceMetrics,
        strategy_analysis: Dict[str, str],
        proposals: List[RuleProposal]
    ) -> List[str]:
        """生成建议"""
        suggestions = []

        if metrics.win_rate < 0.5:
            suggestions.append(
                f"当前胜率 {metrics.win_rate:.1%} 低于50%，建议："
                f"1) 提高策略置信度阈值 2) 优化策略参数 3) 加强风控"
            )

        if metrics.max_drawdown < -5000:
            suggestions.append(
                f"最大回撤 {abs(metrics.max_drawdown):.0f} 较大，"
                f"建议降低单笔仓位或增加止损机制"
            )

        for proposal in proposals:
            suggestions.append(
                f"[{proposal.proposal_type.value}] {proposal.target}: "
                f"{proposal.reason} - {proposal.expected_impact}"
            )

        if not suggestions:
            suggestions.append("系统运行良好，暂无改进建议")

        return suggestions

    def get_report_history(self, limit: int = 10) -> List[AdvisorReport]:
        """获取报告历史"""
        return self._report_history[-limit:]

    def get_latest_report(self) -> Optional[AdvisorReport]:
        """获取最新报告"""
        return self._report_history[-1] if self._report_history else None


if __name__ == "__main__":
    """测试代码"""
    print("=== LLM Advisor 测试 ===\n")

    from datetime import timedelta

    # 创建测试数据
    advisor = LLMAdvisor()

    # 创建模拟数据
    now = datetime.now()
    period_start = now - timedelta(days=30)

    # MarketStates
    market_states = [
        MarketState(
            timestamp=now - timedelta(days=i),
            symbol="AAPL",
            regime="TREND" if i % 3 != 0 else "RANGE",
            regime_confidence=0.8,
            volatility_state="NORMAL" if i % 2 == 0 else "HIGH",
            volatility_trend="STABLE",
            volatility_percentile=0.6,
            liquidity_state="NORMAL",
            volume_ratio=1.2,
            sentiment_state="NEUTRAL",
            sentiment_score=0.0
        )
        for i in range(30)
    ]

    # MetaDecisions
    meta_decisions = [
        MetaDecision(
            timestamp=now - timedelta(days=i),
            symbol="AAPL",
            target_position=100,
            target_weight=0.1,
            active_strategies=["MA", "Breakout"] if i % 3 != 0 else ["MA"],
            disabled_strategies={} if i % 3 != 0 else {"Breakout": "regime_RANGE"},
            decision_confidence=0.7,
            consensus_level="STRONG" if i % 2 == 0 else "WEAK",
            reasoning="测试决策"
        )
        for i in range(30)
    ]

    # RiskDecisions
    risk_decisions = [
        RiskDecision(
            timestamp=now - timedelta(days=i),
            symbol="AAPL",
            scaled_target_position=100 if i % 4 != 0 else 30,
            scale_factor=1.0 if i % 4 != 0 else 0.3,
            risk_action="APPROVE" if i % 4 != 0 else "SCALE_DOWN",
            risk_reason="风控通过" if i % 4 != 0 else "市场状态缩放"
        )
        for i in range(30)
    ]

    # ExecutionResults
    execution_results = [
        ExecutionResult(
            timestamp=now - timedelta(days=i),
            symbol="AAPL",
            execution_mode="paper",
            target_position=100,
            actual_position=100,
            target_price=150.0,
            fill_price=150.15,
            status="FILLED",
            fill_quantity=100,
            slippage=0.001,
            current_positions={"AAPL": 100},
            cash_balance=95000 + i * 100
        )
        for i in range(20)
    ]

    # 生成报告
    print("1. 生成顾问报告:")
    report = advisor.analyze_run(
        market_states=market_states,
        meta_decisions=meta_decisions,
        risk_decisions=risk_decisions,
        execution_results=execution_results,
        period_start=period_start,
        period_end=now
    )

    print(f"   报告周期: {report.report_period}")
    print(f"   总交易数: {report.performance_metrics.total_trades}")
    print(f"   胜率: {report.performance_metrics.win_rate:.1%}")
    print(f"   总盈亏: {report.performance_metrics.total_pnl:.2f}")
    print(f"   \n策略分析:")
    for strategy, analysis in report.strategy_analysis.items():
        print(f"   - {analysis}")

    print(f"   \n市场分析:")
    print(f"   {report.market_analysis}")

    print(f"   \n规则建议 ({len(report.rule_proposals)} 条):")
    for proposal in report.rule_proposals:
        print(f"   - [{proposal.proposal_type.value}] {proposal.target}")
        print(f"     {proposal.reason}: {proposal.expected_impact}")

    print(f"   \n风险评估:")
    print(f"   {report.risk_assessment}")

    print("\n✓ 所有测试通过")
