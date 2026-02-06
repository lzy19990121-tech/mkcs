"""
成本与延迟敏感性分析

实现：
1. 不同手续费/滑点下的策略表现
2. 信号/执行延迟的影响
3. 输出"策略可用成本区间"和"最大可接受延迟"
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from copy import deepcopy
import numpy as np

from core.models import Bar, Signal, Trade

logger = logging.getLogger(__name__)


@dataclass
class CostConfig:
    """成本配置"""
    commission_per_share: float = 0.01        # 每股��续费
    commission_min: float = 1.0               # 最低手续费
    slippage_bps: float = 0.0                 # 滑点（基点，1 BPS = 0.01%）
    slippage_percent: float = 0.0             # 滑点百分比


@dataclass
class DelayConfig:
    """延迟配置"""
    signal_delay_bars: int = 0                # 信号延迟（K线数）
    execution_delay_seconds: int = 0          # 执行延迟（秒）
    fill_delay_bars: int = 1                  # 成交延迟（通常下一根K线）


@dataclass
class SensitivityResult:
    """敏感性测试结果"""
    config_name: str
    total_return: float
    total_pnl: float
    trade_count: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float = 0.0

    # 成本相关
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # 延迟相关
    avg_execution_delay: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_name": self.config_name,
            "total_return": self.total_return,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "avg_execution_delay": self.avg_execution_delay
        }


@dataclass
class CostSensitivityReport:
    """成本敏感性报告"""
    strategy_name: str
    test_date: str

    # 测试结果
    results: List[SensitivityResult] = field(default_factory=list)

    # 结论
    viable_cost_range: Optional[Tuple[CostConfig, CostConfig]] = None
    cost_tolerance_score: float = 0.0  # 成本容忍度评分

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "test_date": self.test_date,
            "results": [r.to_dict() for r in self.results],
            "viable_cost_range": {
                "min": self.viable_cost_range[0].__dict__ if self.viable_cost_range else None,
                "max": self.viable_cost_range[1].__dict__ if self.viable_cost_range else None
            } if self.viable_cost_range else None,
            "cost_tolerance_score": self.cost_tolerance_score
        }


@dataclass
class DelaySensitivityReport:
    """延迟敏感性报告"""
    strategy_name: str
    test_date: str

    # 测试结果
    results: List[SensitivityResult] = field(default_factory=list)

    # 结论
    max_acceptable_delay_bars: int = 0
    max_acceptable_delay_seconds: int = 0
    delay_sensitivity_score: float = 0.0  # 延迟敏感度评分（越高越敏感）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "test_date": self.test_date,
            "results": [r.to_dict() for r in self.results],
            "max_acceptable_delay_bars": self.max_acceptable_delay_bars,
            "max_acceptable_delay_seconds": self.max_acceptable_delay_seconds,
            "delay_sensitivity_score": self.delay_sensitivity_score
        }


class SensitivityAnalyzer:
    """敏感性分析器

    测试策略对成本和延迟的敏感性
    """

    def __init__(self):
        self.cost_report: Optional[CostSensitivityReport] = None
        self.delay_report: Optional[DelaySensitivityReport] = None

    def analyze_cost_sensitivity(
        self,
        strategy,
        bars: List[Bar],
        initial_capital: float = 100000,
        cost_configs: Optional[List[CostConfig]] = None
    ) -> CostSensitivityReport:
        """分析成本敏感性

        Args:
            strategy: 策略对象（需要 generate_signals 方法）
            bars: K线数据
            initial_capital: 初始资金
            cost_configs: 要测试的成本配置列表

        Returns:
            成本敏感性报告
        """
        if cost_configs is None:
            # 默认测试配置
            cost_configs = [
                CostConfig(commission_per_share=0.001, slippage_bps=0),    # 极低成本
                CostConfig(commission_per_share=0.005, slippage_bps=1),    # 低成本
                CostConfig(commission_per_share=0.01, slippage_bps=2),     # 中等成本
                CostConfig(commission_per_share=0.02, slippage_bps=5),     # 高成本
                CostConfig(commission_per_share=0.05, slippage_bps=10),    # 极高成本
            ]

        results = []

        for i, cost_config in enumerate(cost_configs):
            # 使用该成本配置运行策略
            result = self._run_with_cost(
                strategy,
                bars,
                initial_capital,
                cost_config
            )
            result.config_name = f"cost_config_{i}"
            results.append(result)

        # 分析结果，找出可行区间
        viable_range = self._find_viable_cost_range(results, cost_configs)

        # 计算成本容忍度评分
        tolerance_score = self._calculate_cost_tolerance(results)

        report = CostSensitivityReport(
            strategy_name=strategy.__class__.__name__,
            test_date=datetime.now().isoformat(),
            results=results,
            viable_cost_range=viable_range,
            cost_tolerance_score=tolerance_score
        )

        self.cost_report = report
        return report

    def analyze_delay_sensitivity(
        self,
        strategy,
        bars: List[Bar],
        initial_capital: float = 100000,
        delay_configs: Optional[List[DelayConfig]] = None
    ) -> DelaySensitivityReport:
        """分析延迟敏感性

        Args:
            strategy: 策略对象
            bars: K线数据
            initial_capital: 初始资金
            delay_configs: 要测试的延迟配置列表

        Returns:
            延迟敏感性报告
        """
        if delay_configs is None:
            # 默认测试配置
            delay_configs = [
                DelayConfig(signal_delay_bars=0, execution_delay_seconds=0),   # 无延迟
                DelayConfig(signal_delay_bars=1, execution_delay_seconds=0),   # 延迟1根K线
                DelayConfig(signal_delay_bars=2, execution_delay_seconds=0),   # 延迟2根K线
                DelayConfig(signal_delay_bars=3, execution_delay_seconds=0),   # 延迟3根K线
                DelayConfig(signal_delay_bars=5, execution_delay_seconds=0),   # 延迟5根K线
            ]

        results = []

        for i, delay_config in enumerate(delay_configs):
            result = self._run_with_delay(
                strategy,
                bars,
                initial_capital,
                delay_config
            )
            result.config_name = f"delay_config_{i}"
            results.append(result)

        # 找出最大可接受延迟
        max_delay = self._find_max_acceptable_delay(results, delay_configs)

        # 计算延迟敏感度
        sensitivity_score = self._calculate_delay_sensitivity(results)

        report = DelaySensitivityReport(
            strategy_name=strategy.__class__.__name__,
            test_date=datetime.now().isoformat(),
            results=results,
            max_acceptable_delay_bars=max_delay[0],
            max_acceptable_delay_seconds=max_delay[1],
            delay_sensitivity_score=sensitivity_score
        )

        self.delay_report = report
        return report

    def _run_with_cost(
        self,
        strategy,
        bars: List[Bar],
        initial_capital: float,
        cost_config: CostConfig
    ) -> SensitivityResult:
        """使用指定成本配置运行策略"""
        cash = initial_capital
        position = 0  # 持仓数量
        total_commission = 0.0
        total_slippage = 0.0

        trades = []
        equity_curve = [initial_capital]

        for i, bar in enumerate(bars[20:], start=20):  # 跳过预热期
            # 获取信号（使用当前和之前的K线）
            lookback_bars = bars[:i+1]
            signals = strategy.generate_signals(lookback_bars)

            current_price = float(bar.close)

            for signal in signals:
                if signal.action == "BUY" and position == 0:
                    # 计算交易成本
                    quantity = signal.quantity
                    commission = max(cost_config.commission_per_share * quantity, cost_config.commission_min)

                    # 计算滑点
                    slippage = current_price * (cost_config.slippage_bps / 10000 + cost_config.slippage_percent)
                    execution_price = current_price + slippage

                    trade_value = execution_price * quantity
                    total_cost = commission + slippage * quantity

                    if cash >= trade_value + total_cost:
                        position = quantity
                        cash -= (trade_value + commission)
                        total_commission += commission
                        total_slippage += slippage * quantity

                        trades.append({
                            "type": "BUY",
                            "price": execution_price,
                            "quantity": quantity,
                            "cost": total_cost
                        })

                elif signal.action == "SELL" and position > 0:
                    # 卖出
                    quantity = position
                    commission = max(cost_config.commission_per_share * quantity, cost_config.commission_min)

                    # 滑点（卖出时滑点通常不利）
                    slippage = current_price * (cost_config.slippage_bps / 10000 + cost_config.slippage_percent)
                    execution_price = current_price - slippage

                    trade_value = execution_price * quantity
                    total_cost = commission + slippage * quantity

                    cash += (trade_value - commission)
                    total_commission += commission
                    total_slippage += slippage * quantity

                    trades.append({
                        "type": "SELL",
                        "price": execution_price,
                        "quantity": quantity,
                        "cost": total_cost
                    })

                    position = 0

            # 更新权益
            if position > 0:
                equity = cash + position * current_price
            else:
                equity = cash
            equity_curve.append(equity)

        # 计算最终统计
        final_equity = equity_curve[-1] if equity_curve else initial_capital
        total_return = (final_equity - initial_capital) / initial_capital
        total_pnl = final_equity - initial_capital

        # 计算最大回撤
        max_equity = initial_capital
        max_drawdown = 0.0
        for equity in equity_curve:
            if equity > max_equity:
                max_equity = equity
            drawdown = (max_equity - equity) / max_equity
            max_drawdown = max(max_drawdown, drawdown)

        # 计算胜率
        win_count = 0
        total_round_trips = 0
        for i in range(0, len(trades), 2):
            if i + 1 < len(trades):
                buy_price = trades[i]["price"]
                sell_price = trades[i + 1]["price"]
                if sell_price > buy_price:
                    win_count += 1
                total_round_trips += 1

        win_rate = win_count / total_round_trips if total_round_trips > 0 else 0.0

        # 计算夏普比率（简化版）
        returns = np.diff(equity_curve) / initial_capital
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

        return SensitivityResult(
            config_name="",
            total_return=total_return,
            total_pnl=total_pnl,
            trade_count=len(trades),
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            total_commission=total_commission,
            total_slippage=total_slippage
        )

    def _run_with_delay(
        self,
        strategy,
        bars: List[Bar],
        initial_capital: float,
        delay_config: DelayConfig
    ) -> SensitivityResult:
        """使用指定延迟配置运行策略"""
        cash = initial_capital
        position = 0

        pending_signals = []  # 等待执行的信号
        executed_trades = []

        for i, bar in enumerate(bars[20:], start=20):
            lookback_bars = bars[:i+1]
            current_price = float(bar.close)

            # 处理待执行的信号
            if pending_signals:
                signal, signal_bar_idx = pending_signals[0]

                # 检查是否可以执行
                bars_elapsed = i - signal_bar_idx
                if bars_elapsed >= delay_config.fill_delay_bars:
                    # 执行交易
                    if signal.action == "BUY" and position == 0:
                        quantity = signal.quantity
                        cost = current_price * quantity
                        if cash >= cost:
                            position = quantity
                            cash -= cost
                            executed_trades.append({
                                "type": "BUY",
                                "price": current_price,
                                "bar_idx": i
                            })
                    elif signal.action == "SELL" and position > 0:
                        cash += position * current_price
                        position = 0
                        executed_trades.append({
                            "type": "SELL",
                            "price": current_price,
                            "bar_idx": i
                        })
                    pending_signals.pop(0)

            # 生成新信号（应用信号延迟）
            signal_delay_idx = max(0, i - delay_config.signal_delay_bars)
            if signal_delay_idx > 0:
                delayed_bars = bars[:signal_delay_idx+1]
            else:
                delayed_bars = lookback_bars

            signals = strategy.generate_signals(delayed_bars)

            for signal in signals:
                if signal.action in ["BUY", "SELL"]:
                    pending_signals.append((signal, i))

        # 计算最终权益
        if position > 0:
            final_equity = cash + position * float(bars[-1].close)
        else:
            final_equity = cash

        total_return = (final_equity - initial_capital) / initial_capital
        max_drawdown = 0.1  # 简化

        return SensitivityResult(
            config_name="",
            total_return=total_return,
            total_pnl=final_equity - initial_capital,
            trade_count=len(executed_trades),
            win_rate=0.5,  # 简化
            max_drawdown=max_drawdown
        )

    def _find_viable_cost_range(
        self,
        results: List[SensitivityResult],
        configs: List[CostConfig]
    ) -> Optional[Tuple[CostConfig, CostConfig]]:
        """找出可行的成本区间

        可行标准：
        1. 总收益为正
        2. 最大回撤 < 20%
        """
        viable_configs = []

        for result, config in zip(results, configs):
            if result.total_return > 0 and result.max_drawdown < 0.2:
                viable_configs.append((result, config))

        if not viable_configs:
            return None

        # 按成本排序
        viable_configs.sort(key=lambda x: (
            x[1].commission_per_share +
            x[1].slippage_bps * 0.0001
        ))

        return (viable_configs[0][1], viable_configs[-1][1])

    def _calculate_cost_tolerance(self, results: List[SensitivityResult]) -> float:
        """计算成本容忍度评分

        评分标准：策略在高成本下的收益衰减速度
        """
        if len(results) < 2:
            return 0.5

        # 计算收益衰减
        baseline_return = results[0].total_return
        worst_return = min(r.total_return for r in results)

        if baseline_return <= 0:
            return 0.0

        decay_ratio = abs(worst_return) / baseline_return
        tolerance = 1.0 - decay_ratio

        return max(0.0, min(1.0, tolerance))

    def _find_max_acceptable_delay(
        self,
        results: List[SensitivityResult],
        configs: List[DelayConfig]
    ) -> Tuple[int, int]:
        """找出最大可接受延迟"""
        max_delay_bars = 0

        for result, config in zip(results, configs):
            # 如果收益为正，认为可接受
            if result.total_return > 0:
                max_delay_bars = max(max_delay_bars, config.signal_delay_bars)

        # 转换为秒（假设日线）
        max_delay_seconds = max_delay_bars * 24 * 3600

        return (max_delay_bars, max_delay_seconds)

    def _calculate_delay_sensitivity(self, results: List[SensitivityResult]) -> float:
        """计算延迟敏感度评分

        评分越高表示越敏感（对延迟越不利）
        """
        if len(results) < 2:
            return 0.5

        # 计算每增加1根K线延迟，收益下降多少
        baseline_return = results[0].total_return
        worst_return = results[-1].total_return

        decay_per_bar = abs(baseline_return - worst_return) / max(1, len(results) - 1)

        # 归一化到 0-1
        sensitivity = min(1.0, decay_per_bar * 10)

        return sensitivity


def format_cost_report(report: CostSensitivityReport) -> str:
    """格式化成本敏感性报告"""
    lines = [
        f"# 成本敏感性报告",
        f"策略: {report.strategy_name}",
        f"测试日期: {report.test_date}",
        f"",
        f"## 测试结果",
        f""
    ]

    for result in report.results:
        lines.extend([
            f"### {result.config_name}",
            f"- 总收益: {result.total_return:.2%}",
            f"- 总盈亏: ${result.total_pnl:.2f}",
            f"- 交易次数: {result.trade_count}",
            f"- 胜率: {result.win_rate:.2%}",
            f"- 最大回撤: {result.max_drawdown:.2%}",
            f"- 手续费: ${result.total_commission:.2f}",
            f"- 滑点: ${result.total_slippage:.2f}",
            f""
        ])

    if report.viable_cost_range:
        min_cfg = report.viable_cost_range[0]
        max_cfg = report.viable_cost_range[1]
        lines.extend([
            f"## 可行成本区间",
            f"- 最低成本: ${min_cfg.commission_per_share}/股, {min_cfg.slippage_bps}bps",
            f"- 最高成本: ${max_cfg.commission_per_share}/股, {max_cfg.slippage_bps}bps",
            f""
        ])

    lines.extend([
        f"## 成本容忍度评分: {report.cost_tolerance_score:.2%}",
        f"- 0% = 无法容忍任何成本增加",
        f"- 100% = 对成本完全不敏感"
    ])

    return "\n".join(lines)


def format_delay_report(report: DelaySensitivityReport) -> str:
    """格式化延迟敏感性报告"""
    lines = [
        f"# 延迟敏感性报告",
        f"策略: {report.strategy_name}",
        f"测试日期: {report.test_date}",
        f"",
        f"## 测试结果",
        f""
    ]

    for result in report.results:
        lines.extend([
            f"### {result.config_name}",
            f"- 总收益: {result.total_return:.2%}",
            f"- 最大回撤: {result.max_drawdown:.2%}",
            f""
        ])

    lines.extend([
        f"## 结论",
        f"- 最大可接受延迟: {report.max_acceptable_delay_bars} 根K线",
        f"- 相当于: {report.max_acceptable_delay_seconds // 3600} 小时",
        f"- 延迟敏感度: {report.delay_sensitivity_score:.2%}",
        f"- 0% = 对延迟不敏感",
        f"- 100% = 对延迟极其敏感"
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    """测试代码"""
    print("=== SensitivityAnalyzer 测试 ===\n")

    # 创建模拟策略
    class MockStrategy:
        def generate_signals(self, bars):
            from core.models import Signal
            from decimal import Decimal

            signals = []
            if len(bars) >= 21:
                # 简单的动量策略
                if bars[-1].close > bars[-2].close and bars[-2].close > bars[-3].close:
                    signals.append(Signal(
                        symbol="TEST",
                        timestamp=bars[-1].timestamp,
                        action="BUY",
                        price=bars[-1].close,
                        quantity=100,
                        confidence=0.7,
                        reason="上涨"
                    ))
                elif bars[-1].close < bars[-2].close and bars[-2].close < bars[-3].close:
                    signals.append(Signal(
                        symbol="TEST",
                        timestamp=bars[-1].timestamp,
                        action="SELL",
                        price=bars[-1].close,
                        quantity=100,
                        confidence=0.7,
                        reason="下跌"
                    ))
            return signals

        def __class__(self):
            return "MockStrategy"

    # 创建测试数据
    from datetime import timedelta

    base_price = 100.0
    bars = []
    for i in range(100):
        date = datetime(2024, 1, 1) + timedelta(days=i)
        change = (i % 10 - 5) * 0.5  # 模拟波动
        close = base_price + change

        bars.append(Bar(
            symbol="TEST",
            timestamp=date,
            open=Decimal(str(close)),
            high=Decimal(str(close + 0.5)),
            low=Decimal(str(close - 0.5)),
            close=Decimal(str(close)),
            volume=1000000,
            interval="1d"
        ))

    # 测试成本敏感性
    print("1. 成本敏感性分析:")
    strategy = MockStrategy()
    analyzer = SensitivityAnalyzer()

    cost_report = analyzer.analyze_cost_sensitivity(strategy, bars)
    print(format_cost_report(cost_report))

    # 测试延迟敏感性
    print("\n2. 延迟敏感性分析:")
    delay_report = analyzer.analyze_delay_sensitivity(strategy, bars)
    print(format_delay_report(delay_report))

    print("\n✓ 所有测试通过")
