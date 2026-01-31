"""
多策略组合框架

支持多个策略组合使用，加权投票，风险分散
"""

import logging
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from core.models import Bar, Signal, Position, OrderIntent
from skills.strategy.base import Strategy
from skills.strategy.moving_average import MAStrategy
from skills.strategy.ml_strategy import MLStrategy
from skills.risk.base import RiskManager

logger = logging.getLogger(__name__)


class AllocationMethod(Enum):
    """资金分配方法"""
    EQUAL_WEIGHT = "equal"           # 等权重
    PERFORMANCE_BASED = "performance"  # 基于历史表现
    RISK_PARITY = "risk_parity"     # 风险平价
    CUSTOM = "custom"               # 自定义权重


@dataclass
class StrategyConfig:
    """策略配置"""
    strategy: Strategy
    weight: float = 1.0              # 策略权重
    max_position_ratio: float = 0.2  # 单策略最大仓位比例
    enabled: bool = True             # 是否启用
    name: str = ""                   # 策略名称
    allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT


class PortfolioStrategy(Strategy):
    """多策略组合

    整合多个子策略，通过加权投票生成最终信号
    """

    def __init__(
        self,
        strategies: List[StrategyConfig],
        vote_threshold: float = 0.5,
        disagreement_threshold: float = 0.3
    ):
        """初始化多策略组合

        Args:
            strategies: 策略配置列表
            vote_threshold: 投票阈值（权重之和达到此值才执行）
            disagreement_threshold: 意见分歧阈值（分歧过大时不执行）
        """
        self.strategies = strategies
        self.vote_threshold = vote_threshold
        self.disagreement_threshold = disagreement_threshold

        # 验证权重
        total_weight = sum(s.weight for s in strategies if s.enabled)
        if total_weight == 0:
            raise ValueError("总权重不能为0")
        self.total_weight = total_weight

        # 策略名称
        for i, s in enumerate(strategies):
            if not s.name:
                s.name = f"Strategy_{i}"

        logger.info(f"初始化多策略组合: {len(strategies)} 个策略, 总权重={total_weight:.2f}")

    def generate_signals(
        self,
        bars: List[Bar],
        position: Optional[Position] = None
    ) -> List[Signal]:
        """生成组合信号

        Args:
            bars: K 线数据列表
            position: 当前持仓

        Returns:
            组合后的信号列表
        """
        # 收集所有策略的信号
        all_signals = {}  # {symbol: [(strategy_name, signal, weight), ...]}

        for strategy_config in self.strategies:
            if not strategy_config.enabled:
                continue

            try:
                signals = strategy_config.strategy.generate_signals(bars, position)

                for signal in signals:
                    if signal.symbol not in all_signals:
                        all_signals[signal.symbol] = []

                    all_signals[signal.symbol].append({
                        'name': strategy_config.name,
                        'signal': signal,
                        'weight': strategy_config.weight,
                        'confidence': signal.confidence
                    })

            except Exception as e:
                logger.error(f"策略 {strategy_config.name} 生成信号失败: {e}")

        # 合并信号
        combined_signals = []
        for symbol, signal_list in all_signals.items():
            combined = self._combine_signals(symbol, signal_list, bars[-1], position)
            if combined:
                combined_signals.append(combined)

        return combined_signals

    def _combine_signals(
        self,
        symbol: str,
        signal_list: List[Dict],
        current_bar: Bar,
        position: Optional[Position]
    ) -> Optional[Signal]:
        """合并多个策略的信号

        Args:
            symbol: 标的代码
            signal_list: 信号列表
            current_bar: 当前 K 线
            position: 当前持仓

        Returns:
            合并后的信号，或 None（如果不满足执行条件）
        """
        # 按操作类型分组
        buy_votes = []
        sell_votes = []
        hold_votes = []

        for item in signal_list:
            signal = item['signal']
            weight = item['weight']
            confidence = signal.confidence

            weighted_vote = weight * confidence

            if signal.action == 'BUY':
                buy_votes.append(weighted_vote)
            elif signal.action == 'SELL':
                sell_votes.append(weighted_vote)
            else:
                hold_votes.append(weighted_vote)

        buy_score = sum(buy_votes)
        sell_score = sum(sell_votes)
        hold_score = sum(hold_votes)

        # 检查意见分歧
        total_score = buy_score + sell_score + hold_score
        if total_score > 0:
            max_score = max(buy_score, sell_score, hold_score)
            second_max = sorted([buy_score, sell_score, hold_score])[-2]

            if total_score > 0 and (max_score - second_max) / total_score < self.disagreement_threshold:
                logger.info(f"{symbol}: 策略意见分歧过大，不执行")
                return None

        # 决定最终操作
        action = None
        final_confidence = 0
        reasons = []

        if buy_score > sell_score and buy_score > hold_score:
            # 买入信号
            action = 'BUY'
            final_confidence = buy_score / self.total_weight
            reasons = [f"{s['name']}:BUY({s['confidence']:.2f})" for s in signal_list if s['signal'].action == 'BUY']

        elif sell_score > buy_score and sell_score > hold_score:
            # 卖出信号
            action = 'SELL'
            final_confidence = sell_score / self.total_weight
            reasons = [f"{s['name']}:SELL({s['confidence']:.2f})" for s in signal_list if s['signal'].action == 'SELL']

        else:
            # 持有
            return None

        # 检查投票阈值
        if final_confidence < self.vote_threshold:
            logger.info(f"{symbol}: 投票权重不足 ({final_confidence:.2f} < {self.vote_threshold})")
            return None

        # 检查是否已经有持仓（只有卖出时需要持仓）
        if action == 'SELL' and (not position or position.quantity == 0):
            return None

        # 计算数量（基于所有策略的平均值）
        quantities = [s['signal'].quantity for s in signal_list if s['signal'].action == action]
        avg_quantity = int(sum(quantities) / len(quantities)) if quantities else 100

        return Signal(
            symbol=symbol,
            timestamp=current_bar.timestamp,
            action=action,
            price=current_bar.close,
            quantity=avg_quantity,
            confidence=min(final_confidence, 1.0),
            reason=f"组合投票: {', '.join(reasons[:3])}"  # 只显示前3个
        )

    def get_min_bars_required(self) -> int:
        """获取所需的最小 K 线数量"""
        min_bars = 0
        for strategy_config in self.strategies:
            if strategy_config.enabled:
                required = strategy_config.strategy.get_min_bars_required()
                min_bars = max(min_bars, required)
        return min_bars

    def get_strategy_status(self) -> Dict[str, Dict]:
        """获取各策略状态

        Returns:
            策略状态字典 {strategy_name: {...}}
        """
        status = {}
        for s in self.strategies:
            status[s.name] = {
                'enabled': s.enabled,
                'weight': s.weight,
                'max_position_ratio': s.max_position_ratio,
                'type': type(s.strategy).__name__
            }
        return status

    def set_strategy_weight(self, strategy_name: str, weight: float):
        """动态调整策略权重

        Args:
            strategy_name: 策略名称
            weight: 新权重
        """
        for s in self.strategies:
            if s.name == strategy_name:
                old_weight = s.weight
                s.weight = weight
                self.total_weight = self.total_weight - old_weight + weight
                logger.info(f"策略 {strategy_name} 权重: {old_weight:.2f} → {weight:.2f}")
                return
        raise ValueError(f"策略 {strategy_name} 不存在")

    def enable_strategy(self, strategy_name: str):
        """启用策略"""
        for s in self.strategies:
            if s.name == strategy_name:
                if not s.enabled:
                    s.enabled = True
                    self.total_weight += s.weight
                    logger.info(f"启用策略 {strategy_name}")
                return
        raise ValueError(f"策略 {strategy_name} 不存在")

    def disable_strategy(self, strategy_name: str):
        """禁用策略"""
        for s in self.strategies:
            if s.name == strategy_name:
                if s.enabled:
                    s.enabled = False
                    self.total_weight -= s.weight
                    logger.info(f"禁用策略 {strategy_name}")
                return
        raise ValueError(f"策略 {strategy_name} 不存在")


def create_default_portfolio() -> PortfolioStrategy:
    """创建默认的多策略组合

    Returns:
        配置好的 PortfolioStrategy
    """
    from skills.strategy.ml_strategy import RandomForestModel

    strategies = [
        # MA 策略 (保守)
        StrategyConfig(
            strategy=MAStrategy(fast_period=5, slow_period=20),
            weight=1.0,
            max_position_ratio=0.15,
            name="MA_Crossover_Short"
        ),
        StrategyConfig(
            strategy=MAStrategy(fast_period=10, slow_period=30),
            weight=0.8,
            max_position_ratio=0.15,
            name="MA_Crossover_Medium"
        ),

        # ML 策略 (激进)
        StrategyConfig(
            strategy=MLStrategy(
                model=RandomForestModel(n_estimators=100),
                confidence_threshold=0.6
            ),
            weight=1.2,
            max_position_ratio=0.2,
            name="ML_RandomForest"
        ),
    ]

    portfolio = PortfolioStrategy(
        strategies=strategies,
        vote_threshold=0.4,
        disagreement_threshold=0.3
    )

    return portfolio


def create_ml_portfolio(model_paths: Dict[str, str]) -> PortfolioStrategy:
    """创建纯 ML 策略组合

    Args:
        model_paths: 模型路径字典 {strategy_name: model_path}

    Returns:
        配置好的 PortfolioStrategy
    """
    from skills.strategy.ml_strategy import MLStrategy, LSTMModel, RandomForestModel

    strategies = []

    for name, path in model_paths.items():
        # 根据文件扩展名判断模型类型
        if path.endswith('.h5'):
            model = LSTMModel()
        else:
            model = RandomForestModel()

        strategies.append(StrategyConfig(
            strategy=MLStrategy(model=model, confidence_threshold=0.6),
            weight=1.0,
            name=name
        ))

    return PortfolioStrategy(strategies=strategies)


if __name__ == "__main__":
    """测试代码"""
    print("=== 多策略组合测试 ===\n")

    from skills.market_data.mock_source import MockMarketSource
    from datetime import datetime, timedelta

    # 生成测试数据
    source = MockMarketSource(seed=42)
    bars = source.get_bars(
        "AAPL",
        datetime.now() - timedelta(days=100),
        datetime.now(),
        "1d"
    )

    print(f"生成 {len(bars)} 根 K 线\n")

    # 创建默认组合
    portfolio = create_default_portfolio()

    # 显示策略状态
    print("策略状态:")
    status = portfolio.get_strategy_status()
    for name, info in status.items():
        print(f"  {name}:")
        print(f"    启用: {info['enabled']}")
        print(f"    权重: {info['weight']}")
        print(f"    类型: {info['type']}")

    # 生成信号
    print(f"\n生成组合信号 (最小K线: {portfolio.get_min_bars_required()})...")
    signals = portfolio.generate_signals(bars, position=None)

    print(f"\n生成 {len(signals)} 个组合信号:")
    for signal in signals:
        print(f"  {signal.symbol}: {signal.action} @ ${signal.price} (置信度: {signal.confidence:.2%})")
        print(f"    原因: {signal.reason}")

    # 测试动态调整权重
    print("\n测试动态调整权重...")
    portfolio.set_strategy_weight("MA_Crossover_Short", 0.5)
    print(f"调整后总权重: {portfolio.total_weight:.2f}")

    print("\n✓ 测试完成")
