"""
系统级回归与验收测试 - Phase 8

完整测试所有组件，边界条件，性能
"""

import sys
import time
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List

from core.models import Bar
from core.schema import MarketState, AlphaOpinion, MetaDecision, RiskDecision, ExecutionResult
from skills.strategy.alpha_base import MAStrategy, BreakoutStrategy, MLStrategy
from skills.brain.meta_strategy import MetaStrategyAgent
from skills.risk.risk_gate_agent import RiskGateAgent
from skills.execution.agent import ExecutionAgent
from skills.llm.advisor import LLMAdvisor
from skills.visualization.explainability import ExplainabilityEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_bars(symbol: str, days: int, trend: float = 0.5) -> List[Bar]:
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


class TestResults:
    """测试结果收集"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name: str):
        self.passed += 1
        logger.info(f"✓ {test_name}")

    def add_fail(self, test_name: str, reason: str):
        self.failed += 1
        self.errors.append((test_name, reason))
        logger.error(f"✗ {test_name}: {reason}")

    def summary(self) -> bool:
        total = self.passed + self.failed
        logger.info(f"\n测试完成: {self.passed}/{total} 通过")
        if self.errors:
            logger.error("失败的测试:")
            for name, reason in self.errors:
                logger.error(f"  - {name}: {reason}")
        return self.failed == 0


def test_alpha_strategies(results: TestResults) -> bool:
    """测试 L2 Alpha 策略"""
    logger.info("\n[测试] L2 Alpha 策略")

    try:
        bars = create_bars("AAPL", 50)

        # 测试 MA 策略
        ma = MAStrategy()
        opinion = ma.analyze(bars, MarketState(
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
        ))

        if opinion.direction in [-1, 0, 1]:
            results.add_pass("MA策略输出方向")
        else:
            results.add_fail("MA策略输出方向", f"无效方向: {opinion.direction}")

        if 0 <= opinion.strength <= 1:
            results.add_pass("MA策略强度范围")
        else:
            results.add_fail("MA策略强度范围", f"强度超出范围: {opinion.strength}")

        if 0 <= opinion.confidence <= 1:
            results.add_pass("MA策略置信度范围")
        else:
            results.add_fail("MA策略置信度范围", f"置信度超出范围: {opinion.confidence}")

        # 测试 Breakout 策略
        breakout = BreakoutStrategy()
        opinion = breakout.analyze(bars, MarketState(
            timestamp=datetime.now(),
            symbol="AAPL",
            regime="RANGE",
            regime_confidence=0.7,
            volatility_state="NORMAL",
            volatility_trend="STABLE",
            volatility_percentile=0.5,
            liquidity_state="NORMAL",
            volume_ratio=1.0,
            sentiment_state="NEUTRAL",
            sentiment_score=0.0
        ))

        if opinion.is_disabled:
            results.add_pass("Breakout策略震荡市禁用")
        else:
            results.add_fail("Breakout策略震荡市禁用", "应该被禁用")

        # 测试 ML 策略
        ml = MLStrategy()
        opinion = ml.analyze(bars[:30], MarketState(
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
        ))

        # 数据不足应该被禁用
        if len(bars[:30]) < ml.get_min_bars_required():
            if opinion.is_disabled:
                results.add_pass("ML策略数据不足禁用")
            else:
                results.add_fail("ML策略数据不足禁用", "应该因数据不足被禁用")

    except Exception as e:
        results.add_fail("Alpha策略测试", str(e))
        return False

    return True


def test_meta_strategy(results: TestResults) -> bool:
    """测试 L4 Meta 决策"""
    logger.info("\n[测试] L4 Meta 决策")

    try:
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

        opinions = [
            AlphaOpinion("MA", datetime.now(), "AAPL", 1, 0.8, 0.7, "daily"),
            AlphaOpinion("Breakout", datetime.now(), "AAPL", 1, 0.6, 0.6, "swing"),
            AlphaOpinion("ML", datetime.now(), "AAPL", 1, 0.5, 0.5, "intraday"),
        ]

        meta = MetaStrategyAgent(max_position_weight=0.2)
        decision = meta.decide(market_state, opinions, 0, 100000)

        if decision.target_position >= 0:
            results.add_pass("Meta决策仓位非负")
        else:
            results.add_fail("Meta决策仓位非负", f"仓位为负: {decision.target_position}")

        if 0 <= decision.target_weight <= 0.2:
            results.add_pass("Meta决策权重范围")
        else:
            results.add_fail("Meta决策权重范围", f"权重超出范围: {decision.target_weight}")

        if "MA" in decision.active_strategies:
            results.add_pass("Meta决策包含启用策略")
        else:
            results.add_fail("Meta决策包含启用策略", "未包含MA策略")

        # 测试冲突处理
        conflict_opinions = [
            AlphaOpinion("MA", datetime.now(), "AAPL", 1, 0.8, 0.7, "daily"),
            AlphaOpinion("Breakout", datetime.now(), "AAPL", -1, 0.6, 0.6, "swing"),
        ]
        decision = meta.decide(market_state, conflict_opinions, 0, 100000)

        if decision.consensus_level in ["STRONG", "WEAK", "NONE"]:
            results.add_pass("Meta决策共识级别")
        else:
            results.add_fail("Meta决策共识级别", f"无效共识级别: {decision.consensus_level}")

    except Exception as e:
        results.add_fail("Meta决策测试", str(e))
        return False

    return True


def test_risk_gate(results: TestResults) -> bool:
    """测试 Risk Gate"""
    logger.info("\n[测试] Risk Gate")

    try:
        risk_gate = RiskGateAgent(initial_capital=100000)

        # 正常市场
        normal_state = MarketState(
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

        meta_decision = MetaDecision(
            timestamp=datetime.now(),
            symbol="AAPL",
            target_position=200,
            target_weight=0.2,
            active_strategies=["MA", "Breakout"],
            disabled_strategies={},
            decision_confidence=0.7,
            consensus_level="STRONG"
        )

        risk_decision = risk_gate.check(meta_decision, normal_state, {})

        if risk_decision.risk_action in ["APPROVE", "SCALE_DOWN", "PAUSE", "DISABLE"]:
            results.add_pass("风控动作有效性")
        else:
            results.add_fail("风控动作有效性", f"无效动作: {risk_decision.risk_action}")

        if risk_decision.scale_factor >= 0:
            results.add_pass("风控缩放系数非负")
        else:
            results.add_fail("风控缩放系数非负", f"缩放系数为负: {risk_decision.scale_factor}")

        # 危机模式
        crisis_state = MarketState(
            timestamp=datetime.now(),
            symbol="AAPL",
            regime="CRISIS",
            regime_confidence=0.9,
            volatility_state="EXTREME",
            volatility_trend="RISING",
            volatility_percentile=0.99,
            liquidity_state="FROZEN",
            volume_ratio=0.3,
            sentiment_state="PANIC",
            sentiment_score=-0.9
        )

        risk_decision = risk_gate.check(meta_decision, crisis_state, {})

        if risk_decision.risk_action == "DISABLE":
            results.add_pass("危机模式风控禁止")
        else:
            results.add_fail("危机模式风控禁止", f"应该是DISABLE，实际是: {risk_decision.risk_action}")

    except Exception as e:
        results.add_fail("Risk Gate测试", str(e))
        return False

    return True


def test_execution(results: TestResults) -> bool:
    """测试 L1 Execution"""
    logger.info("\n[测试] L1 Execution")

    try:
        executor = ExecutionAgent(mode="paper", initial_capital=100000)

        risk_decision = RiskDecision(
            timestamp=datetime.now(),
            symbol="AAPL",
            scaled_target_position=100,
            scale_factor=1.0,
            risk_action="APPROVE",
            risk_reason="风控通过"
        )

        result = executor.execute(risk_decision, current_price=150)

        if result.status in ["FILLED", "SKIPPED", "FAILED", "PARTIAL"]:
            results.add_pass("执行状态有效性")
        else:
            results.add_fail("执行状态有效性", f"无效状态: {result.status}")

        if result.execution_mode == "paper":
            results.add_pass("执行模式正确")
        else:
            results.add_fail("执行模式正确", f"模式错误: {result.execution_mode}")

        # 测试资金不足
        large_decision = RiskDecision(
            timestamp=datetime.now(),
            symbol="AAPL",
            scaled_target_position=10000,
            scale_factor=1.0,
            risk_action="APPROVE",
            risk_reason="风控通过"
        )

        result = executor.execute(large_decision, current_price=150)

        if result.status == "FAILED":
            results.add_pass("资金不足检测")
        else:
            results.add_fail("资金不足检测", f"应该失败，实际状态: {result.status}")

    except Exception as e:
        results.add_fail("Execution测试", str(e))
        return False

    return True


def test_boundary_conditions(results: TestResults) -> bool:
    """测试边界条件"""
    logger.info("\n[测试] 边界条件")

    try:
        meta = MetaStrategyAgent()

        # 空观点列表
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

        decision = meta.decide(market_state, [], 0, 100000)

        if decision.target_position == 0:
            results.add_pass("空观点列表处理")
        else:
            results.add_fail("空观点列表处理", f"应该空仓，实际: {decision.target_position}")

        # 所有策略禁用
        opinions = [
            AlphaOpinion("MA", datetime.now(), "AAPL", 0, 0, 0, "daily", is_disabled=True, disabled_reason="test"),
            AlphaOpinion("Breakout", datetime.now(), "AAPL", 0, 0, 0, "swing", is_disabled=True, disabled_reason="test"),
        ]

        decision = meta.decide(market_state, opinions, 0, 100000)

        if len(decision.active_strategies) == 0:
            results.add_pass("全禁用策略处理")
        else:
            results.add_fail("全禁用策略处理", f"应该无启用策略，实际: {decision.active_strategies}")

        # 零资金
        decision = meta.decide(market_state, opinions, 0, 0)

        if decision.target_position == 0:
            results.add_pass("零资金处理")
        else:
            results.add_fail("零资金处理", f"应该零仓位，实际: {decision.target_position}")

    except Exception as e:
        results.add_fail("边界条件测试", str(e))
        return False

    return True


def test_performance(results: TestResults) -> bool:
    """测试性能"""
    logger.info("\n[测试] 性能")

    try:
        bars = create_bars("AAPL", 100)
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

        # 创建组件
        ma = MAStrategy()
        breakout = BreakoutStrategy()
        ml = MLStrategy()
        meta = MetaStrategyAgent()
        risk_gate = RiskGateAgent()
        executor = ExecutionAgent()

        # 性能测试
        iterations = 100
        start_time = time.time()

        for i in range(iterations):
            # Alpha 分析
            opinions = [
                ma.analyze(bars, market_state),
                breakout.analyze(bars, market_state),
                ml.analyze(bars, market_state)
            ]

            # Meta 决策
            meta_decision = meta.decide(market_state, opinions, 0, 100000)

            # 风控
            risk_decision = risk_gate.check(meta_decision, market_state, {})

            # 执行
            executor.execute(risk_decision, 150)

        elapsed = time.time() - start_time
        avg_time = elapsed / iterations * 1000  # ms

        logger.info(f"  {iterations} 次迭代耗时: {elapsed:.2f}s, 平均: {avg_time:.2f}ms")

        if avg_time < 100:  # 每次决策应小于100ms
            results.add_pass(f"性能要求 (<100ms)")
        else:
            results.add_fail("性能要求", f"平均耗时 {avg_time:.2f}ms 超过100ms")

    except Exception as e:
        results.add_fail("性能测试", str(e))
        return False

    return True


def test_data_flow(results: TestResults) -> bool:
    """测试完整数据流"""
    logger.info("\n[测试] 完整数据流")

    try:
        bars = create_bars("AAPL", 50)

        # 创建组件
        strategies = [MAStrategy(), BreakoutStrategy(), MLStrategy()]
        meta = MetaStrategyAgent()
        risk_gate = RiskGateAgent()
        executor = ExecutionAgent()
        engine = ExplainabilityEngine()

        # 多个时间步的模拟
        for i in range(5):
            market_state = MarketState(
                timestamp=datetime.now() + timedelta(hours=i),
                symbol="AAPL",
                regime="TREND" if i % 2 == 0 else "RANGE",
                regime_confidence=0.8,
                volatility_state="NORMAL",
                volatility_trend="STABLE",
                volatility_percentile=0.6,
                liquidity_state="NORMAL",
                volume_ratio=1.2,
                sentiment_state="NEUTRAL",
                sentiment_score=0.0
            )

            # L2: Alpha
            opinions = [s.analyze(bars, market_state) for s in strategies]

            # L4: Meta
            meta_decision = meta.decide(market_state, opinions, 0, 100000)

            # Risk Gate
            risk_decision = risk_gate.check(meta_decision, market_state, {})

            # L1: Execution
            execution_result = executor.execute(risk_decision, float(bars[-1].close), market_state)

            # Explainability
            explanation = engine.explain_decision(
                market_state, opinions, meta_decision, risk_decision, execution_result
            )

            # 验证数据连续性
            if execution_result.symbol != market_state.symbol:
                results.add_fail(f"数据流一致性 (step {i})", "标的不匹配")
                return False

        results.add_pass("完整数据流")

    except Exception as e:
        results.add_fail("完整数据流测试", str(e))
        return False

    return True


def test_llm_advisor(results: TestResults) -> bool:
    """测试 LLM Advisor"""
    logger.info("\n[测试] LLM Advisor")

    try:
        advisor = LLMAdvisor()

        # 创建测试数据
        now = datetime.now()
        market_states = [
            MarketState(
                timestamp=now - timedelta(days=i),
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
            for i in range(10)
        ]

        meta_decisions = [
            MetaDecision(
                timestamp=now - timedelta(days=i),
                symbol="AAPL",
                target_position=100,
                target_weight=0.1,
                active_strategies=["MA"],
                disabled_strategies={},
                decision_confidence=0.7,
                consensus_level="STRONG"
            )
            for i in range(10)
        ]

        risk_decisions = [
            RiskDecision(
                timestamp=now - timedelta(days=i),
                symbol="AAPL",
                scaled_target_position=100,
                scale_factor=1.0,
                risk_action="APPROVE",
                risk_reason="风控通过"
            )
            for i in range(10)
        ]

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
                cash_balance=95000
            )
            for i in range(10)
        ]

        report = advisor.analyze_run(
            market_states, meta_decisions, risk_decisions, execution_results,
            now - timedelta(days=10), now
        )

        if report.summary:
            results.add_pass("LLM Advisor 生成报告")
        else:
            results.add_fail("LLM Advisor 生成报告", "报告为空")

        if len(report.strategy_analysis) > 0:
            results.add_pass("LLM Advisor 策略分析")
        else:
            results.add_fail("LLM Advisor 策略分析", "无策略分析")

    except Exception as e:
        results.add_fail("LLM Advisor测试", str(e))
        return False

    return True


def run_all_tests() -> bool:
    """运行所有测试"""
    print("="*60)
    print("MKCS 系统级回归与验收测试")
    print("="*60)

    results = TestResults()

    # 运行测试套件
    test_alpha_strategies(results)
    test_meta_strategy(results)
    test_risk_gate(results)
    test_execution(results)
    test_boundary_conditions(results)
    test_performance(results)
    test_data_flow(results)
    test_llm_advisor(results)

    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    success = results.summary()

    if success:
        print("\n✓ 所有测试通过 - 系统验收合格")
        return True
    else:
        print("\n✗ 部分测试失败 - 需要修复")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
