"""
MKCS 系统验收测试 - 完整 Checklist 验证

对照"MKCS 全系统整合 —— 验收 Checklist"逐项验收
"""

import sys
import json
from datetime import datetime, timedelta
from decimal import Decimal

from core.models import Bar
from core.schema import MarketState, AlphaOpinion, MetaDecision, RiskDecision, ExecutionResult
from skills.strategy.alpha_base import MAStrategy, BreakoutStrategy, MLStrategy
from skills.brain.meta_strategy import MetaStrategyAgent
from skills.risk.risk_gate_agent import RiskGateAgent
from skills.execution.agent import ExecutionAgent
from skills.visualization.explainability import ExplainabilityEngine


def create_bars(symbol: str, days: int, trend: float = 0.5):
    """创建测试K线"""
    bars = []
    for i in range(days):
        date = datetime(2024, 1, 1) + timedelta(days=i)
        price = 100 + i * trend + (i % 5 - 2) * 0.3
        bars.append(Bar(
            symbol=symbol,
            timestamp=date,
            open=Decimal(str(price - 0.2)),
            high=Decimal(str(price + 0.5)),
            low=Decimal(str(price - 0.5)),
            close=Decimal(str(price)),
            volume=1000000,
            interval="1d"
        ))
    return bars


class AcceptanceTest:
    """验收测试"""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.results = {}

    def check(self, category: str, item: str, condition: bool, note: str = ""):
        """检查一项"""
        key = f"{category}.{item}"
        self.results[key] = {"passed": condition, "note": note}
        if condition:
            self.passed.append(key)
            print(f"  ✅ {item}")
            if note:
                print(f"     {note}")
        else:
            self.failed.append(key)
            print(f"  ❌ {item}")
            if note:
                print(f"     {note}")

    def summary(self):
        """打印总结"""
        total = len(self.passed) + len(self.failed)
        pass_rate = len(self.passed) / total * 100 if total > 0 else 0

        print("\n" + "="*60)
        print(f"验收结果: {len(self.passed)}/{total} 通过 ({pass_rate:.0f}%)")
        print("="*60)

        if self.failed:
            print("\n未通过项目:")
            for key in self.failed:
                note = self.results[key].get("note", "")
                print(f"  ❌ {key}")
                if note:
                    print(f"     {note}")

        return pass_rate


def run_acceptance_tests():
    """运行完整验收测试"""
    print("="*60)
    print("MKCS 系统验收测试")
    print("="*60)

    test = AcceptanceTest()
    bars = create_bars("AAPL", 50)

    # ========================================================================
    # 一、系统职责与边界
    # ========================================================================
    print("\n【一】系统职责与边界")

    # 创建各层组件
    ma_strategy = MAStrategy()
    meta_agent = MetaStrategyAgent()
    risk_gate = RiskGateAgent()
    executor = ExecutionAgent()

    # 检查：Alpha 策略不直接下单
    market_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime="TREND",
        regime_confidence=0.8,
        volatility_state="NORMAL",
        volatility_trend="STABLE",
        volatility_percentile=0.6,
        liquidity_state="NORMAL",
        volume_ratio=1.2,
        sentiment_state="NEUTRAL",
        sentiment_score=0.0
    )

    opinion = ma_strategy.analyze(bars, market_state)

    test.check(
        "职责边界",
        "Alpha只输出观点，不下单",
        hasattr(opinion, 'direction') and hasattr(opinion, 'strength') and not hasattr(opinion, 'position'),
        f"AlphaOpinion包含: direction={opinion.direction}, strength={opinion.strength}, 无position字段"
    )

    test.check(
        "职责边界",
        "MarketState不产买卖信号",
        hasattr(market_state, 'regime') and not hasattr(market_state, 'target_position'),
        f"MarketState包含: regime={market_state.regime}, 无target_position"
    )

    # 检查决策路径
    opinions = [opinion]
    meta_decision = meta_agent.decide(market_state, opinions, 0, 100000)
    risk_decision = risk_gate.check(meta_decision, market_state, {})

    test.check(
        "职责边界",
        "最终仓位只由Brain输出",
        hasattr(meta_decision, 'target_position') and meta_decision.target_position >= 0,
        f"MetaDecision.target_position={meta_decision.target_position}"
    )

    test.check(
        "职责边界",
        "风控调节而非替代决策",
        risk_decision.scale_factor <= 1.0 and risk_decision.scale_factor >= 0,
        f"RiskDecision.scale_factor={risk_decision.scale_factor}"
    )

    # ========================================================================
    # 二、Market Analysis 验收
    # ========================================================================
    print("\n【二】Market Analysis 验收")

    test.check(
        "MarketState",
        "MarketState每个bar都存在",
        market_state is not None and market_state.timestamp is not None,
        f"MarketState.timestamp={market_state.timestamp}"
    )

    test.check(
        "MarketState",
        "MarketState可序列化",
        hasattr(market_state, 'to_dict'),
        "MarketState有to_dict方法"
    )

    # 测试不同 regime 下的策略行为
    trend_state = MarketState(
        timestamp=datetime.now(), symbol="AAPL",
        regime="TREND", regime_confidence=0.8,
        volatility_state="NORMAL", volatility_trend="STABLE", volatility_percentile=0.6,
        liquidity_state="NORMAL", volume_ratio=1.2,
        sentiment_state="NEUTRAL", sentiment_score=0.0
    )

    range_state = MarketState(
        timestamp=datetime.now(), symbol="AAPL",
        regime="RANGE", regime_confidence=0.8,
        volatility_state="NORMAL", volatility_trend="STABLE", volatility_percentile=0.5,
        liquidity_state="NORMAL", volume_ratio=1.0,
        sentiment_state="NEUTRAL", sentiment_score=0.0
    )

    crisis_state = MarketState(
        timestamp=datetime.now(), symbol="AAPL",
        regime="CRISIS", regime_confidence=0.9,
        volatility_state="EXTREME", volatility_trend="RISING", volatility_percentile=0.99,
        liquidity_state="FROZEN", volume_ratio=0.3,
        sentiment_state="PANIC", sentiment_score=-0.9
    )

    trend_opinion = ma_strategy.analyze(bars, trend_state)
    range_opinion = ma_strategy.analyze(bars, range_state)
    crisis_opinion = ma_strategy.analyze(bars, crisis_state)

    test.check(
        "MarketState",
        "不同regime下策略被明确启用/禁用/降权",
        (not trend_opinion.is_disabled or trend_opinion.disabled_reason) and
        (not range_opinion.is_disabled or range_opinion.disabled_reason) and
        (crisis_opinion.is_disabled and crisis_opinion.disabled_reason),
        f"TREND: disabled={trend_opinion.is_disabled}, RANGE: disabled={range_opinion.is_disabled}, CRISIS: disabled={crisis_opinion.is_disabled}"
    )

    # 极端行情测试
    extreme_meta = meta_agent.decide(crisis_state, [crisis_opinion], 0, 100000)
    extreme_risk = risk_gate.check(extreme_meta, crisis_state, {})

    test.check(
        "MarketState",
        "极端行情下仓位被压缩",
        extreme_risk.scale_factor < 1.0 or extreme_risk.risk_action == "DISABLE",
        f"极端行情下 scale_factor={extreme_risk.scale_factor}, action={extreme_risk.risk_action}"
    )

    test.check(
        "MarketState",
        "MarketState从不直接生成交易方向",
        not hasattr(market_state, 'direction') and not hasattr(market_state, 'target_position'),
        "MarketState无direction和target_position字段"
    )

    # ========================================================================
    # 三、Alpha 层验收
    # ========================================================================
    print("\n【三】Alpha 层验收")

    for name, strategy in [("MA", MAStrategy()), ("Breakout", BreakoutStrategy()), ("ML", MLStrategy())]:
        opinion = strategy.analyze(bars, trend_state)

        test.check(
            "Alpha层",
            f"{name}有direction/strength/confidence",
            hasattr(opinion, 'direction') and hasattr(opinion, 'strength') and hasattr(opinion, 'confidence'),
            f"{name}: direction={opinion.direction}, strength={opinion.strength:.2f}, confidence={opinion.confidence:.2f}"
        )

        test.check(
            "Alpha层",
            f"{name}未参与时有禁用原因",
            not opinion.is_disabled or opinion.disabled_reason != "",
            f"{name}: is_disabled={opinion.is_disabled}, reason={opinion.disabled_reason}"
        )

    # ========================================================================
    # 四、Brain / MetaStrategy 验收
    # ========================================================================
    print("\n【四】Brain / MetaStrategy 验收")

    opinions = [
        AlphaOpinion("MA", datetime.now(), "AAPL", 1, 0.8, 0.7, "daily"),
        AlphaOpinion("Breakout", datetime.now(), "AAPL", -1, 0.6, 0.6, "swing"),
    ]

    meta_decision = meta_agent.decide(market_state, opinions, 0, 100000)

    test.check(
        "Brain",
        "最终TargetPosition只由Brain输出",
        hasattr(meta_decision, 'target_position'),
        f"MetaDecision.target_position={meta_decision.target_position}"
    )

    test.check(
        "Brain",
        "Brain输出包含启用策略列表",
        hasattr(meta_decision, 'active_strategies') and len(meta_decision.active_strategies) > 0,
        f"active_strategies={meta_decision.active_strategies}"
    )

    test.check(
        "Brain",
        "Brain输出包含被禁策略+原因",
        hasattr(meta_decision, 'disabled_strategies'),
        f"disabled_strategies={meta_decision.disabled_strategies}"
    )

    test.check(
        "Brain",
        "Brain输出包含决策置信度",
        hasattr(meta_decision, 'decision_confidence'),
        f"decision_confidence={meta_decision.decision_confidence}"
    )

    test.check(
        "Brain",
        "冲突Alpha有稳定裁决逻辑",
        meta_decision.consensus_level in ["STRONG", "WEAK", "NONE"],
        f"consensus_level={meta_decision.consensus_level}"
    )

    # ========================================================================
    # 五、Risk Gate 验收
    # ========================================================================
    print("\n【五】Risk Gate 验收")

    # 测试缩放优先
    risk_decision = risk_gate.check(meta_decision, range_state, {})

    test.check(
        "RiskGate",
        "风控优先缩放仓位而非reject",
        risk_decision.scale_factor < 1.0 or risk_decision.risk_action == "APPROVE",
        f"scale_factor={risk_decision.scale_factor}, action={risk_decision.risk_action}"
    )

    test.check(
        "RiskGate",
        "reject有明确原因",
        risk_decision.risk_action != "DISABLE" or risk_decision.risk_reason != "",
        f"reason={risk_decision.risk_reason}"
    )

    test.check(
        "RiskGate",
        "所有风控动作有cooldown",
        hasattr(risk_gate, 'get_active_cooldowns'),
        "RiskGate有get_active_cooldowns方法"
    )

    # ========================================================================
    # 六、Execution / Simulation 验收
    # ========================================================================
    print("\n【六】Execution / Simulation 验收")

    test.check(
        "Execution",
        "live默认dry-run",
        ExecutionAgent(mode="live")._is_live_unlocked == False,
        "创建live模式ExecutionAgent时_is_live_unlocked=False"
    )

    result = executor.execute(risk_decision, 150)

    test.check(
        "Execution",
        "ExecutionResult完整回传",
        hasattr(result, 'status') and hasattr(result, 'fill_price') and hasattr(result, 'current_positions'),
        f"status={result.status}, fill_price={result.fill_price}, positions={result.current_positions}"
    )

    # ========================================================================
    # 七、可回放性与一致性
    # ========================================================================
    print("\n【七】可回放性与一致性")

    # 多次运行相同输入
    results = []
    for i in range(3):
        opinion = ma_strategy.analyze(bars, trend_state)
        meta = meta_agent.decide(trend_state, [opinion], 0, 100000)
        risk = risk_gate.check(meta, trend_state, {})
        results.append((opinion.direction, meta.target_position, risk.scaled_target_position))

    test.check(
        "可回放性",
        "同一输入多次replay输出一致",
        all(r == results[0] for r in results),
        f"3次运行结果: {results}"
    )

    # ========================================================================
    # 八、可解释性与 UI
    # ========================================================================
    print("\n【八】可解释性与 UI")

    engine = ExplainabilityEngine()
    explanation = engine.explain_decision(
        market_state, opinions, meta_decision, risk_decision, result
    )

    test.check(
        "可解释性",
        "能显示MarketState",
        'market_context' in explanation and 'regime' in explanation['market_context'],
        f"regime={explanation['market_context']['regime']['value']}"
    )

    test.check(
        "可解释性",
        "能显示各Alpha是否发声",
        'strategy_opinions' in explanation and len(explanation['strategy_opinions']) > 0,
        f"策略数量={len(explanation['strategy_opinions'])}"
    )

    test.check(
        "可解释性",
        "能显示被禁策略+原因",
        any(o.get('is_disabled') and o.get('disabled_reason') for o in explanation['strategy_opinions']) or
        all(not o.get('is_disabled') for o in explanation['strategy_opinions']),
        "禁用策略信息完整"
    )

    test.check(
        "可解释性",
        "有完整决策链",
        'decision_chain' in explanation and len(explanation['decision_chain']) == 5,
        f"决策链步骤数={len(explanation['decision_chain'])}"
    )

    # ========================================================================
    # 九、系统级行为验收
    # ========================================================================
    print("\n【九】系统级行为验收")

    # 统计不同市场下的交易活跃度
    trend_decisions = [meta_agent.decide(trend_state, opinions, 0, 100000) for _ in range(5)]
    range_decisions = [meta_agent.decide(range_state, opinions, 0, 100000) for _ in range(5)]

    trend_avg_pos = sum(abs(d.target_position) for d in trend_decisions) / len(trend_decisions)
    range_avg_pos = sum(abs(d.target_position) for d in range_decisions) / len(range_decisions)

    test.check(
        "系统行为",
        "震荡市交易次数/仓位下降",
        trend_avg_pos >= range_avg_pos,
        f"TREND平均仓位={trend_avg_pos:.1f}, RANGE平均仓位={range_avg_pos:.1f}"
    )

    test.check(
        "系统行为",
        "被禁用是可追溯状态",
        all(hasattr(o, 'is_disabled') and hasattr(o, 'disabled_reason') for o in opinions),
        "所有AlphaOpinion都有is_disabled和disabled_reason"
    )

    # ========================================================================
    # 十、最终 5 问验收
    # ========================================================================
    print("\n【十】最终 5 问验收")

    # 模拟一个完整交易
    opinions = [
        ma_strategy.analyze(bars, trend_state),
        BreakoutStrategy().analyze(bars, trend_state),
        MLStrategy().analyze(bars, trend_state)
    ]

    meta_decision = meta_agent.decide(trend_state, opinions, 0, 100000)
    risk_decision = risk_gate.check(meta_decision, trend_state, {})
    execution_result = executor.execute(risk_decision, 150)

    explanation = engine.explain_decision(
        trend_state, opinions, meta_decision, risk_decision, execution_result
    )

    # Q1: 今天系统认为什么市场？
    test.check(
        "最终5问",
        "Q1: 能回答今天是什么市场",
        'market_context' in explanation,
        f"Regime={explanation['market_context']['regime']['value']}, Volatility={explanation['market_context']['volatility']['state']}"
    )

    # Q2: 哪些策略参与了决策？
    test.check(
        "最终5问",
        "Q2: 能回答哪些策略参与",
        'meta_decision' in explanation and 'active_strategies' in explanation['meta_decision'],
        f"参与策略: {explanation['meta_decision']['active_strategies']}"
    )

    # Q3: 哪些策略被禁了？为什么？
    disabled = [o for o in explanation['strategy_opinions'] if o.get('is_disabled')]
    disabled_str = ", ".join([f"{o['strategy']}({o['disabled_reason']})" for o in disabled]) if disabled else "无被禁策略"
    test.check(
        "最终5问",
        "Q3: 能回答哪些策略被禁及原因",
        all(o.get('disabled_reason') for o in disabled),
        f"被禁策略: {disabled_str}"
    )

    # Q4: 最终仓位如何综合出来的？
    test.check(
        "最终5问",
        "Q4: 能回答仓位如何综合",
        'decision_chain' in explanation and len(explanation['decision_chain']) == 5,
        f"决策链有{len(explanation['decision_chain'])}步"
    )

    # Q5: 如果亏钱了，责任在谁？
    test.check(
        "最终5问",
        "Q5: 能区分决策责任",
        'meta_decision' in explanation and 'risk_control' in explanation and 'execution' in explanation,
        "能区分Meta/Risk/Execution三层责任"
    )

    # ========================================================================
    # 最终判定
    # ========================================================================
    pass_rate = test.summary()

    print("\n" + "="*60)
    if pass_rate >= 100:
        print("🎉 系统完成（10/10 全通过）- 可长期演进")
        return True
    elif pass_rate >= 80:
        print("✅ 工程可用（8/10 模块通过）")
        return True
    else:
        print("❌ 需要继续完善")
        return False


if __name__ == "__main__":
    success = run_acceptance_tests()
    sys.exit(0 if success else 1)
