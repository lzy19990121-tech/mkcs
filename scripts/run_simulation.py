"""
MKCS 模拟交易脚本

使用 5 层架构系统进行模拟交易
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# 导入核心组件
from agent.live_runner import LiveTrader, LiveTradingConfig, TradingMode
from skills.market_data.yahoo_source import YahooFinanceSource
from skills.strategy.moving_average import MAStrategy
from skills.risk.basic_risk import BasicRiskManager
from broker.paper import PaperBroker

# 导入 5 层架构组件
from analysis.online.risk_metrics_collector import RiskMetricsCollector
from analysis.online.risk_state_machine import RiskStateMachine
from analysis.pipeline_optimizer_v2 import PipelineOptimizerV2, PipelineConfig
from analysis.optimization_risk_proxies import RiskProxyCalculator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MKCSSimulation:
    """MKCS 模拟交易系统"""

    def __init__(
        self,
        symbols: list = None,
        initial_cash: float = 100000,
        duration_minutes: int = 30
    ):
        """初始化模拟交易系统

        Args:
            symbols: 交易标的列表
            initial_cash: 初始资金
            duration_minutes: 模拟时长（分钟）
        """
        self.symbols = symbols or ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        self.initial_cash = initial_cash
        self.duration_minutes = duration_minutes

        # 统计数据
        self.trades_count = 0
        self.signals_count = 0
        self.risk_events_count = 0
        self.start_time = None
        self.end_time = None

        logger.info("=" * 60)
        logger.info("MKCS 模拟交易系统初始化")
        logger.info("=" * 60)
        logger.info(f"交易标的: {', '.join(self.symbols)}")
        logger.info(f"初始资金: ${initial_cash:,.2f}")
        logger.info(f"模拟时长: {duration_minutes} 分钟")

        self._initialize_components()

    def _initialize_components(self):
        """初始化各层组件"""

        # ========== L1: 数据层 ==========
        logger.info("\n[L1] 初始化数据层...")
        self.data_source = YahooFinanceSource(enable_cache=True)

        # ========== L2: 策略层 ==========
        logger.info("[L2] 初始化策略层...")
        self.strategies = {}
        strategy_configs = [
            {"name": "MA_Fast", "fast_period": 5, "slow_period": 15},
            {"name": "MA_Medium", "fast_period": 10, "slow_period": 30},
            {"name": "MA_Slow", "fast_period": 20, "slow_period": 50},
        ]

        for config in strategy_configs:
            strategy_id = f"ma_{config['fast_period']}_{config['slow_period']}"
            self.strategies[strategy_id] = MAStrategy(
                fast_period=config['fast_period'],
                slow_period=config['slow_period']
            )
            logger.info(f"  - {config['name']}: MA({config['fast_period']}, {config['slow_period']})")

        # ========== L3: 风控层 ==========
        logger.info("[L3] 初始化风控层...")
        self.risk_manager = BasicRiskManager(
            max_position_ratio=0.3,  # 单个标的最大仓位 30%
            max_positions=10,        # 最大持仓数量
            max_daily_loss_ratio=0.05  # 最大日损失 5%
        )
        self.risk_manager.set_capital(self.initial_cash)

        # 在线风险监控
        self.risk_collectors = {}
        self.risk_state_machines = {}
        for strategy_id in self.strategies:
            self.risk_collectors[strategy_id] = RiskMetricsCollector(
                strategy_id=strategy_id
            )
            self.risk_state_machines[strategy_id] = RiskStateMachine(
                strategy_id=strategy_id
            )

        # ========== L4: 执行层 ==========
        logger.info("[L4] 初始化执行层...")
        self.broker = PaperBroker(initial_cash=Decimal(str(self.initial_cash)))
        logger.info(f"  - Paper Broker: 初始资金 ${self.initial_cash:,.2f}")

        # ========== L5: 优化层 ==========
        logger.info("[L5] 初始化优化层...")
        self.pipeline_config = PipelineConfig(
            enable_gating=True,
            enable_optimizer=True,
            smooth_penalty_lambda=2.0
        )
        self.optimizer = PipelineOptimizerV2(
            strategy_ids=list(self.strategies.keys()),
            config=self.pipeline_config
        )
        logger.info(f"  - Pipeline Optimizer: gating=True, optimizer=True, lambda=2.0")

        # 当前权重
        self.current_weights = {sid: 1.0/len(self.strategies) for sid in self.strategies}

        logger.info("\n初始化完成！")

    def fetch_market_data(self, symbol: str, days: int = 60):
        """获取市场数据

        Args:
            symbol: 标的代码
            days: 获取天数

        Returns:
            List[Bar]: 市场数据
        """
        try:
            # 使用 UTC 时间避免时区问题
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)

            bars = self.data_source.get_bars(
                symbol=symbol,
                start=start_date,
                end=end_date,
                interval="1d"
            )
            return bars
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None

    def generate_signals(self, symbol: str, bars):
        """生成交易信号

        Args:
            symbol: 标的代码
            bars: 市场数据列表

        Returns:
            dict: 各策略的信号
        """
        signals = {}

        if not bars or len(bars) < 50:
            return signals

        current_price = float(bars[-1].close)

        for strategy_id, strategy in self.strategies.items():
            try:
                # 简化处理：使用MA交叉逻辑生成信号
                # 获取收盘价列表
                closes = [float(b.close) for b in bars]

                # 计算MA
                if hasattr(strategy, 'fast_period') and hasattr(strategy, 'slow_period'):
                    fast_period = strategy.fast_period
                    slow_period = strategy.slow_period

                    if len(closes) >= slow_period:
                        fast_ma = sum(closes[-fast_period:]) / fast_period
                        slow_ma = sum(closes[-slow_period:]) / slow_period

                        # 金叉买入，死叉卖出
                        if fast_ma > slow_ma and closes[-2] <= closes[-1]:
                            action = 'BUY'
                            strength = (fast_ma - slow_ma) / slow_ma
                        elif fast_ma < slow_ma and closes[-2] >= closes[-1]:
                            action = 'SELL'
                            strength = (slow_ma - fast_ma) / slow_ma
                        else:
                            action = 'HOLD'
                            strength = 0
                    else:
                        action = 'HOLD'
                        strength = 0

                    signal = {
                        'action': action,
                        'strength': strength,
                        'confidence': min(abs(strength) * 10, 1.0),
                        'fast_ma': fast_ma if len(closes) >= fast_period else 0,
                        'slow_ma': slow_ma if len(closes) >= slow_period else 0,
                        'price': current_price
                    }
                else:
                    signal = {'action': 'HOLD', 'strength': 0, 'confidence': 0.5}

                signals[strategy_id] = signal

                # 更新风险指标
                self.risk_collectors[strategy_id].update(
                    current_price=current_price,
                    current_position=0,  # 简化处理
                    timestamp=datetime.now()
                )

                # 更新状态机
                state = self.risk_state_machines[strategy_id].update(
                    signal_value=signal.get('strength', 0),
                    risk_level=signal.get('confidence', 0.5)
                )

                if signal.get('action') != 'HOLD':
                    self.signals_count += 1
                    logger.debug(f"  {strategy_id}: {signal['action']} @ {current_price:.2f}")

            except Exception as e:
                logger.warning(f"策略 {strategy_id} 生成信号失败: {e}")
                signals[strategy_id] = {'action': 'HOLD', 'strength': 0}

        return signals

    def execute_trade(self, symbol: str, action: str, quantity: int = 100):
        """执行交易

        Args:
            symbol: 标的代码
            action: 交易方向 (BUY/SELL)
            quantity: 数量

        Returns:
            bool: 是否成功
        """
        try:
            from core.models import OrderIntent, Side

            side = Side.BUY if action == 'BUY' else Side.SELL

            # 创建订单意图
            intent = OrderIntent(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type='MARKET'
            )

            # 提交订单
            order = self.broker.submit_order(intent)

            if order:
                # 立即撮合
                from decimal import Decimal
                fills = self.broker.fill_orders({symbol: Decimal("150.00")})

                if fills:
                    self.trades_count += 1
                    logger.info(f"  ✅ 订单成交: {action} {symbol} x{quantity}")
                    return True
                else:
                    logger.warning(f"  ⏸ 订单未成交: {action} {symbol} x{quantity}")
                    return False
            else:
                logger.warning(f"  ❌ 订单被拒绝: {action} {symbol} x{quantity}")
                self.risk_events_count += 1
                return False

        except Exception as e:
            logger.warning(f"  ❌ 交易失败: {e}")
            return False

    def run_cycle(self):
        """运行一个交易周期"""
        logger.info("\n" + "=" * 60)
        logger.info(f"交易周期: {datetime.now().strftime('%H:%M:%S')}")
        logger.info("=" * 60)

        total_signals = {}

        # 获取所有标的数据并生成信号
        for symbol in self.symbols:
            logger.info(f"\n📊 {symbol}")

            # 获取数据
            bars = self.fetch_market_data(symbol)
            if bars is None or len(bars) < 50:
                logger.warning(f"  数据不足，跳过")
                continue

            current_price = float(bars[-1].close)
            logger.info(f"  当前价格: ${current_price:.2f}")

            # 生成信号
            signals = self.generate_signals(symbol, bars)
            total_signals[symbol] = signals

            # 汇总信号
            buy_signals = sum(1 for s in signals.values() if s.get('action') == 'BUY')
            sell_signals = sum(1 for s in signals.values() if s.get('action') == 'SELL')

            logger.info(f"  信号统计: BUY={buy_signals}, SELL={sell_signals}, HOLD={len(signals)-buy_signals-sell_signals}")

        # 使用优化器决定最终权重（简化处理）
        # 这里我们使用信号投票来决定交易
        for symbol, signals in total_signals.items():
            buy_votes = sum(1 for s in signals.values() if s.get('action') == 'BUY')
            sell_votes = sum(1 for s in signals.values() if s.get('action') == 'SELL')

            # 简单投票规则
            if buy_votes >= 2:  # 至少 2 个策略买入
                self.execute_trade(symbol, 'BUY', quantity=100)
            elif sell_votes >= 2:  # 至少 2 个策略卖出
                self.execute_trade(symbol, 'SELL', quantity=100)

        # 显示账户状态
        self._print_account_status()

    def _print_account_status(self):
        """打印账户状态"""
        cash = self.broker.cash
        positions = self.broker.positions

        logger.info("\n💰 账户状态:")
        logger.info(f"  现金: ${float(cash):,.2f}")

        if positions:
            logger.info(f"  持仓:")
            for symbol, pos in positions.items():
                logger.info(f"    {symbol}: {pos.quantity} 股 @ ${float(pos.avg_price):.2f}")

        # 计算总权益
        total_equity = float(cash)
        for pos in positions.values():
            total_equity += float(pos.quantity) * float(pos.avg_price)

        logger.info(f"  总权益: ${total_equity:,.2f}")

    def run(self):
        """运行模拟交易"""
        logger.info("\n" + "=" * 60)
        logger.info("开始模拟交易")
        logger.info("=" * 60)

        self.start_time = datetime.now()
        end_time = self.start_time + timedelta(minutes=self.duration_minutes)

        cycle = 0
        while datetime.now() < end_time:
            cycle += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"周期 #{cycle}")
            logger.info(f"{'='*60}")

            self.run_cycle()

            # 等待下一个周期
            remaining = (end_time - datetime.now()).total_seconds()
            if remaining > 60:  # 如果剩余时间超过 1 分钟，等待 1 分钟
                logger.info(f"\n⏰ 等待 60 秒后进行下一周期...")
                time.sleep(60)
            else:
                logger.info(f"\n⏰ 模拟即将结束，剩余 {int(remaining)} 秒")
                time.sleep(min(remaining, 10))

        self.end_time = datetime.now()
        self._print_summary()

    def _print_summary(self):
        """打印交易总结"""
        duration = (self.end_time - self.start_time).total_seconds() / 60

        logger.info("\n" + "=" * 60)
        logger.info("模拟交易总结")
        logger.info("=" * 60)
        logger.info(f"运行时长: {duration:.1f} 分钟")
        logger.info(f"交易次数: {self.trades_count}")
        logger.info(f"信号数量: {self.signals_count}")
        logger.info(f"风控事件: {self.risk_events_count}")

        self._print_account_status()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="MKCS 模拟交易系统")
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL'],
        help='交易标的列表'
    )
    parser.add_argument(
        '--cash',
        type=float,
        default=100000,
        help='初始资金'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='模拟时长（分钟）'
    )

    args = parser.parse_args()

    # 创建并运行模拟
    simulation = MKCSSimulation(
        symbols=args.symbols,
        initial_cash=args.cash,
        duration_minutes=args.duration
    )

    try:
        simulation.run()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  用户中断，正在退出...")
        simulation.end_time = datetime.now()
        simulation._print_summary()


if __name__ == "__main__":
    main()
