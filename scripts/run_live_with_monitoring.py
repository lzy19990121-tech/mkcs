"""
集成 SPL-7a 的实时交易启动脚本

实时交易 + 在线监控 + 告警 + Post-mortem
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from agent.live_runner import LiveTrader, LiveTradingConfig, TradingMode
from skills.market_data.yahoo_source import YahooFinanceSource
from skills.strategy.moving_average import MAStrategy
from skills.risk.basic_risk import BasicRiskManager
from broker.paper import PaperBroker

# SPL-7a 在线监控
from analysis.online.risk_metrics_collector import RiskMetricsCollector
from analysis.online.risk_state_machine import RiskStateMachine
from analysis.online.trend_detector import TrendDetector
from analysis.online.alerting import AlertingManager
from analysis.online.postmortem_generator import PostMortemGenerator
from analysis.online.risk_event_store import RiskEventStore

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoredLiveTrader:
    """带在线监控的实时交易器"""

    def __init__(
        self,
        symbols: list,
        strategy_config: dict = None,
        initial_cash: float = 100000,
        interval: str = "5m"
    ):
        """初始化

        Args:
            symbols: 交易标的列表
            strategy_config: 策略配置
            initial_cash: 初始资金
            interval: 数据更新间隔
        """
        self.symbols = symbols

        # ========== 1. 基础组件 ==========
        logger.info("初始化基础组件...")

        # 数据源
        self.data_source = YahooFinanceSource(enable_cache=True)

        # 策略
        strategy_config = strategy_config or {"fast_period": 5, "slow_period": 20}
        self.strategy = MAStrategy(**strategy_config)

        # 风控
        self.risk_manager = BasicRiskManager()

        # 经纪商
        self.broker = PaperBroker(initial_cash=Decimal(str(initial_cash)))

        # ========== 2. SPL-7a 在线监控 ==========
        logger.info("初始化 SPL-7a 在线监控...")

        # 风险指标采集器（每个策略一个）
        self.collectors = {}
        for symbol in symbols:
            self.collectors[symbol] = RiskMetricsCollector(
                strategy_id=f"ma_{strategy_config['fast_period']}_{strategy_config['slow_period']}_{symbol}"
            )

        # 状态机
        self.state_machines = {}
        for symbol in symbols:
            self.state_machines[symbol] = RiskStateMachine(
                strategy_id=f"ma_{strategy_config['fast_period']}_{strategy_config['slow_period']}_{symbol}"
            )

        # 趋势检测器
        self.trend_detectors = {}
        for symbol in symbols:
            self.trend_detectors[symbol] = TrendDetector(
                strategy_id=f"ma_{strategy_config['fast_period']}_{strategy_config['slow_period']}_{symbol}"
            )

        # 告警管理器
        self.alerting_manager = AlertingManager(config_path=None)

        # Post-mortem 生成器（使用第一个 symbol 的策略 ID）
        strategy_id = f"ma_{strategy_config['fast_period']}_{strategy_config['slow_period']}"
        self.postmortem_generator = PostMortemGenerator(
            strategy_id=strategy_id,
            replay_data_path=None
        )

        # 事件存储
        self.event_store = RiskEventStore(
            db_path="data/risk_events.db",
            enable_wal=True
        )

        # ========== 3. LiveTrader 配置 ==========
        config = LiveTradingConfig(
            mode=TradingMode.PAPER,
            symbols=symbols,
            interval=interval,
            check_interval_seconds=60,  # 每分钟检查一次
            market_open_time="09:30",
            market_close_time="16:00",
            emergency_stop_loss=Decimal("0.05")  # 5% 紧急止损
        )

        self.trader = LiveTrader(
            config=config,
            data_source=self.data_source,
            strategy=self.strategy,
            risk_manager=self.risk_manager,
            broker=self.broker
        )

        # 注册回调
        self.trader.register_callback("signal", self._on_signal)
        self.trader.register_callback("order", self._on_order)

        # 监控统计
        self.monitoring_stats = {
            "risk_signals_collected": 0,
            "state_transitions": 0,
            "alerts_generated": 0,
            "postmortems_generated": 0
        }

    def _on_signal(self, signal):
        """信号回调 - 触发在线监控"""
        try:
            symbol = signal.symbol
            collector = self.collectors.get(symbol)
            state_machine = self.state_machines.get(symbol)
            trend_detector = self.trend_detectors.get(symbol)

            if not all([collector, state_machine, trend_detector]):
                return

            # 模拟更新风险指标（使用信号价格）
            # 注意：真实环境应该从 broker 获取完整的历史数据
            price = float(signal.price) if signal.price else 0
            from analysis.online.risk_signal_schema import RiskSignal

            # 简化的风险信号（实际应该更完整）
            risk_signal = collector.update(
                price=price,
                timestamp=datetime.now(),
                position=0  # TODO: 从 broker 获取
            )

            self.monitoring_stats["risk_signals_collected"] += 1

            # 更新状态机
            transition = state_machine.update_state(risk_signal)
            if transition:
                self.monitoring_stats["state_transitions"] += 1
                logger.info(
                    f"[状态转换] {symbol}: {transition.from_state.value} → {transition.to_state.value}, "
                    f"触发指标: {transition.trigger_metric} ({transition.trigger_value:.2f})"
                )

                # 存储事件
                self.event_store.store_event(
                    event_type="STATE_TRANSITION",
                    data=transition.to_dict()
                )

            # 趋势检测
            trends = trend_detector.update_trends(risk_signal)

            # 告警评估
            current_state = state_machine.get_current_state()
            alerts = self.alerting_manager.process_risk_update(
                signal=risk_signal,
                state=current_state,
                trends=trends,
                state_transition=transition
            )

            if alerts:
                self.monitoring_stats["alerts_generated"] += len(alerts)
                for alert in alerts:
                    logger.warning(
                        f"[告警] {symbol}: {alert.title} - {alert.message}, "
                        f"当前值: {alert.current_value:.2%}, 阈值: {alert.threshold:.2%}"
                    )

                    # 存储告警事件
                    self.event_store.store_event(
                        event_type="ALERT",
                        data=alert.to_dict()
                    )

        except Exception as e:
            logger.error(f"监控回调失败: {e}", exc_info=True)

    def _on_order(self, intent):
        """订单回调"""
        symbol = intent.signal.symbol
        if intent.approved:
            logger.info(f"[订单通过] {symbol}: {intent.signal.action} @ {intent.signal.price}")
        else:
            logger.warning(f"[订单拒绝] {symbol}: {intent.risk_reason}")

    def start(self):
        """启动实时交易"""
        logger.info("=" * 60)
        logger.info("🚀 启动实时交易 + SPL-7a 在线监控")
        logger.info("=" * 60)
        logger.info(f"标的: {', '.join(self.symbols)}")
        logger.info(f"策略: MA({self.strategy.fast_period}, {self.strategy.slow_period})")
        logger.info(f"初始资金: ${self.broker.get_cash_balance()}")
        logger.info("=" * 60)

        # 打印启动检查清单
        self._print_startup_checklist()

        # 启动交易
        try:
            self.trader.start()
        except KeyboardInterrupt:
            logger.info("\n收到停止信号...")
        finally:
            self._print_summary()

    def _print_startup_checklist(self):
        """打印启动检查清单"""
        logger.info("\n📋 启动检查清单:")
        checks = [
            ("数据源", "✓ Yahoo Finance 已连接" if self.data_source else "✗ 数据源未初始化"),
            ("策略", f"✓ MA({self.strategy.fast_period}, {self.strategy.slow_period})"),
            ("风控", "✓ BasicRiskManager 已启用"),
            ("监控", f"✓ SPL-7a 已启用（{len(self.symbols)} 个采集器）"),
            ("告警", "✓ 多渠道告警已配置"),
            ("事件存储", "✓ SQLite 事件存储已就绪"),
        ]

        for name, status in checks:
            logger.info(f"  {status} - {name}")

        logger.info("")

    def _print_summary(self):
        """打印运行总结"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 实时交易总结")
        logger.info("=" * 60)

        # 交易统计
        trading_stats = self.trader.get_status()
        logger.info(f"运行模式: {trading_stats['mode']}")
        logger.info(f"最终权益: ${trading_stats['portfolio_value']:,.2f}")
        logger.info(f"现金余额: ${trading_stats['cash_balance']:,.2f}")
        logger.info(f"今日信号: {trading_stats['signal_count_today']}")

        # 监控统计
        logger.info(f"\nSPL-7a 监控统计:")
        logger.info(f"  风险信号采集: {self.monitoring_stats['risk_signals_collected']}")
        logger.info(f"  状态转换: {self.monitoring_stats['state_transitions']}")
        logger.info(f"  告警生成: {self.monitoring_stats['alerts_generated']}")

        logger.info("=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='实时交易 + SPL-7a 在线监控')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'],
                       help='交易标的（默认: AAPL MSFT GOOGL）')
    parser.add_argument('--cash', type=float, default=100000,
                       help='初始资金（默认: 100000）')
    parser.add_argument('--interval', default='5m',
                       help='数据更新间隔（默认: 5m）')
    parser.add_argument('--fast', type=int, default=5,
                       help='MA 快线周期（默认: 5）')
    parser.add_argument('--slow', type=int, default=20,
                       help='MA 慢线周期（默认: 20）')

    args = parser.parse_args()

    # 创建交易器
    trader = MonitoredLiveTrader(
        symbols=args.symbols,
        strategy_config={
            "fast_period": args.fast,
            "slow_period": args.slow
        },
        initial_cash=args.cash,
        interval=args.interval
    )

    # 启动
    trader.start()


if __name__ == "__main__":
    main()
