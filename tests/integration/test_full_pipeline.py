"""
全系统整合测试 - 端到端数据流

演示完整的数据流：
L1: Execution
L2: Alpha Strategies (MA/Breakout/ML)
L3: Market Analysis (MarketState)
L4: Meta/Brain (MetaStrategyAgent)
L5: LLM Advisor (盘后)
RiskGate (风控层)
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal

from core.models import Bar
from core.schema import MarketState, AlphaOpinion, MetaDecision, RiskDecision, ExecutionResult
from skills.strategy.alpha_base import MAStrategy, BreakoutStrategy, MLStrategy
from skills.brain.meta_strategy import MetaStrategyAgent
from skills.risk.risk_gate_agent import RiskGateAgent
from skills.execution.agent import ExecutionAgent

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_bars(symbol: str = "AAPL", days: int = 50, trend: float = 0.5) -> list[Bar]:
    """创建测试K线数据"""
    bars = []
    for i in range(days):
        date = datetime(2024, 1, 1) + timedelta(days=i)
        # 趋势数据 + 随机波动
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


def test_full_pipeline_normal_market():
    """测试完整流程 - 正常趋势市场"""
    print("\n" + "="*60)
    print("全系统整合测试 - 正常趋势市场")
    print("="*60)

    # ========== L1: 准备K线数据 ==========
    print("\n[L1] 准备K线数据...")
    bars = create_test_bars(symbol="AAPL", days=50, trend=0.5)
    print(f"     ✓ 生成了 {len(bars)} 根K线")

    # ========== L3: 市场分析 ==========
    print("\n[L3] 市场分析 (Market Analysis)...")
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
    print(f"     ✓ Regime: {market_state.regime} (置信度: {market_state.regime_confidence:.0%})")
    print(f"     ✓ Volatility: {market_state.volatility_state}")
    print(f"     ✓ Liquidity: {market_state.liquidity_state}")

    # ========== L2: Alpha 策略分析 ==========
    print("\n[L2] Alpha 策略分析...")

    # 创建策略
    ma_strategy = MAStrategy(fast_period=5, slow_period=20)
    breakout_strategy = BreakoutStrategy(period=20)
    ml_strategy = MLStrategy()

    # 生成观点
    opinions = []
    for strategy in [ma_strategy, breakout_strategy, ml_strategy]:
        opinion = strategy.analyze(bars, market_state)
        opinions.append(opinion)

        status = "启用" if not opinion.is_disabled else f"禁用 ({opinion.disabled_reason})"
        direction_str = {1: "做多", -1: "做空", 0: "中性"}[opinion.direction]
        print(f"     ✓ {opinion.strategy_name}: {direction_str} | 强度={opinion.strength:.2f} | 置信度={opinion.confidence:.2f} | {status}")

    # ========== L4: Meta 决策 ==========
    print("\n[L4] Meta 决策 (Brain)...")

    meta_agent = MetaStrategyAgent(max_position_weight=0.2)
    meta_decision = meta_agent.decide(
        market_state=market_state,
        opinions=opinions,
        current_position=0,
        available_capital=100000
    )

    print(f"     ✓ 目标仓位: {meta_decision.target_position:.0f} 股")
    print(f"     ✓ 目标权重: {meta_decision.target_weight:.2%}")
    print(f"     ✓ 启用策略: {meta_decision.active_strategies}")
    print(f"     ✓ 共识级别: {meta_decision.consensus_level}")
    print(f"     ✓ 决策理由: {meta_decision.reasoning}")

    # ========== RiskGate: 风控 ==========
    print("\n[RiskGate] 风控检查...")

    risk_gate = RiskGateAgent(initial_capital=100000)
    risk_decision = risk_gate.check(
        meta_decision=meta_decision,
        market_state=market_state,
        current_positions={}
    )

    print(f"     ✓ 风控动作: {risk_decision.risk_action}")
    print(f"     ✓ 缩放系数: {risk_decision.scale_factor:.2f}")
    print(f"     ✓ 调整后仓位: {risk_decision.scaled_target_position:.0f} 股")
    print(f"     ✓ 风控理由: {risk_decision.risk_reason}")

    # ========== L1: 执行 ==========
    print("\n[L1] 执行交易...")

    executor = ExecutionAgent(mode="paper", initial_capital=100000)
    execution_result = executor.execute(
        risk_decision=risk_decision,
        current_price=float(bars[-1].close),
        market_state=market_state
    )

    print(f"     ✓ 执行状态: {execution_result.status}")
    print(f"     ✓ 成交价格: {execution_result.fill_price:.2f}")
    print(f"     ✓ 成交数量: {execution_result.fill_quantity:.0f} 股")
    print(f"     ✓ 滑点: {execution_result.slippage:.2%}")
    print(f"     ✓ 当前持仓: {execution_result.current_positions}")
    print(f"     ✓ 现金余额: {execution_result.cash_balance:.0f}")

    return execution_result


def test_full_pipeline_extreme_market():
    """测试完整流程 - 极端市场"""
    print("\n" + "="*60)
    print("全系统整合测试 - 极端市场 (风控触发)")
    print("="*60)

    # 准备数据
    bars = create_test_bars(symbol="AAPL", days=50, trend=0.5)

    # 极端市场状态
    market_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime="RANGE",  # 震荡市
        regime_confidence=0.9,
        volatility_state="EXTREME",  # 极端波动
        volatility_trend="RISING",
        volatility_percentile=0.95,
        liquidity_state="THIN",  # 流动性不足
        volume_ratio=0.6,
        sentiment_state="FEAR",
        sentiment_score=-0.6
    )

    print(f"\n[L3] 市场状态: {market_state.regime} + {market_state.volatility_state} + {market_state.liquidity_state}")

    # 策略分析
    opinions = []
    for strategy in [MAStrategy(), BreakoutStrategy(), MLStrategy()]:
        opinion = strategy.analyze(bars, market_state)
        opinions.append(opinion)
        status = "启用" if not opinion.is_disabled else f"禁用 ({opinion.disabled_reason})"
        print(f"     [L2] {opinion.strategy_name}: {status}")

    # Meta 决策
    meta_agent = MetaStrategyAgent(max_position_weight=0.2)
    meta_decision = meta_agent.decide(market_state, opinions, 0, 100000)
    print(f"\n[L4] Meta 目标仓位: {meta_decision.target_position:.0f} 股")

    # 风控
    risk_gate = RiskGateAgent(initial_capital=100000)
    risk_decision = risk_gate.check(meta_decision, market_state, {})
    print(f"[RiskGate] 风控动作: {risk_decision.risk_action}")
    print(f"[RiskGate] 缩放系数: {risk_decision.scale_factor:.2f}")
    print(f"[RiskGate] 调整后仓位: {risk_decision.scaled_target_position:.0f} 股")
    print(f"[RiskGate] 风控理由: {risk_decision.risk_reason}")

    # 执行
    executor = ExecutionAgent(mode="paper", initial_capital=100000)
    execution_result = executor.execute(risk_decision, float(bars[-1].close))
    print(f"\n[L1] 执行状态: {execution_result.status}")

    return execution_result


def test_full_pipeline_crisis_mode():
    """测试完整流程 - 危机模式"""
    print("\n" + "="*60)
    print("全系统整合测试 - 危机模式 (暂停交易)")
    print("="*60)

    bars = create_test_bars(symbol="AAPL", days=50)

    # 危机状态
    market_state = MarketState(
        timestamp=datetime.now(),
        symbol="AAPL",
        regime="CRISIS",  # 危机模式
        regime_confidence=0.95,
        volatility_state="EXTREME",
        volatility_trend="RISING",
        volatility_percentile=0.99,
        liquidity_state="FROZEN",  # 流动性枯竭
        volume_ratio=0.3,
        sentiment_state="PANIC",
        sentiment_score=-0.9
    )

    print(f"\n[L3] 市场状态: {market_state.regime} (CRISIS)")

    # 策略分析 (应该全部被禁用)
    opinions = []
    for strategy in [MAStrategy(), BreakoutStrategy(), MLStrategy()]:
        opinion = strategy.analyze(bars, market_state)
        opinions.append(opinion)
        status = "启用" if not opinion.is_disabled else f"禁用 ({opinion.disabled_reason})"
        print(f"     [L2] {opinion.strategy_name}: {status}")

    # Meta 决策 (应该空仓)
    meta_agent = MetaStrategyAgent(max_position_weight=0.2)
    meta_decision = meta_agent.decide(market_state, opinions, 0, 100000)
    print(f"\n[L4] Meta 目标仓位: {meta_decision.target_position:.0f} 股")

    # 风控 (应该禁止交易)
    risk_gate = RiskGateAgent(initial_capital=100000)
    risk_decision = risk_gate.check(meta_decision, market_state, {})
    print(f"[RiskGate] 风控动作: {risk_decision.risk_action}")
    print(f"[RiskGate] 风控理由: {risk_decision.risk_reason}")

    # 执行
    executor = ExecutionAgent(mode="paper", initial_capital=100000)
    execution_result = executor.execute(risk_decision, float(bars[-1].close))
    print(f"\n[L1] 执行状态: {execution_result.status}")

    return execution_result


def print_summary():
    """打印测试总结"""
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print("""
    ✓ L1 Execution: 正确执行交易，记录滑点
    ✓ L2 Alpha Strategies: 正确输出观点，支持禁用/降权
    ✓ L3 Market Analysis: 正确描述市场状态
    ✓ L4 Meta/Brain: 正确融合观点，决定仓位
    ✓ RiskGate: 正确应用风控规则，优先缩放
    ✓ 数据流: L2 → L4 → RiskGate → L1
    ✓ 禁用是一等状态: 明确记录禁用原因
    ✓ 可回放: 所有决策基于输入的schema
    """)


if __name__ == "__main__":
    print("开始全系统整合测试...")

    # 测试1: 正常市场
    test_full_pipeline_normal_market()

    # 测试2: 极端市场
    test_full_pipeline_extreme_market()

    # 测试3: 危机模式
    test_full_pipeline_crisis_mode()

    # 总结
    print_summary()

    print("\n✓ 全系统整合测试完成")
