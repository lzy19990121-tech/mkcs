"""
实时交易模式

从回测模式切换到实时数据模式，支持连接真实市场数据进行实盘/模拟交易
"""

import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Callable, TYPE_CHECKING
import pytz

if TYPE_CHECKING:
    from analysis.online.risk_monitor import RiskMonitor
from dataclasses import dataclass, field
from enum import Enum

from core.models import Bar, Signal, OrderIntent, Trade, Position
from core.context import RunContext
from skills.market_data.base import MarketDataSource
from skills.market_data.yahoo_source import YahooFinanceSource
from skills.strategy.base import Strategy
from skills.risk.base import RiskManager
from broker.paper import PaperBroker
from events.event_log import EventLogger, Event

# SPL-7a Risk Monitor
try:
    from analysis.online.risk_monitor import RiskMonitor
    RISK_MONITOR_AVAILABLE = True
except ImportError:
    RISK_MONITOR_AVAILABLE = False
    logger.warning("SPL-7a RiskMonitor 不可用")

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """交易模式"""
    BACKTEST = "backtest"      # 回测模式
    PAPER = "paper"            # 模拟盘（实时数据，模拟交易）
    LIVE = "live"              # 实盘（实时数据，真实交易）


@dataclass
class LiveTradingConfig:
    """实时交易配置"""
    mode: TradingMode = TradingMode.PAPER
    symbols: List[str] = field(default_factory=list)
    interval: str = "1m"                      # 数据更新频率
    check_interval_seconds: int = 60          # 检查间隔（秒）
    market_open_time: str = "09:30"           # 市场开盘时间
    market_close_time: str = "16:00"          # 市场收盘时间
    timezone: str = "America/New_York"        # 时区
    enable_after_hours: bool = False          # 是否允许盘后交易
    max_daily_signals: int = 10               # 每日最大信号数
    emergency_stop_loss: Decimal = Decimal("0.05")  # 紧急止损比例

    # SPL-7a 配置
    enable_risk_monitor: bool = True          # 是否启用 SPL-7a 监控
    risk_monitor_output: str = "data/live_monitoring"  # 监控输出目录
    alerts_config: Optional[str] = None       # 告警配置文件路径


class LiveTrader:
    """实时交易器

    支持从实时数据源获取数据，执行策略和风控，进行模拟或实盘交易
    """

    def __init__(
        self,
        config: LiveTradingConfig,
        data_source: Optional[MarketDataSource] = None,
        strategy: Optional[Strategy] = None,
        risk_manager: Optional[RiskManager] = None,
        broker: Optional[PaperBroker] = None,
        event_logger: Optional[EventLogger] = None
    ):
        """初始化实时交易器

        Args:
            config: 实时交易配置
            data_source: 市场数据源（默认 YahooFinanceSource）
            strategy: 交易策略
            risk_manager: 风控管理器
            broker: 经纪商（默认 PaperBroker）
            event_logger: 事件日志记录器
        """
        self.config = config
        self.data_source = data_source or YahooFinanceSource()
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.broker = broker or PaperBroker()
        self.event_logger = event_logger or EventLogger()

        # ========== SPL-7a Risk Monitor ==========
        self.risk_monitor = None
        if config.enable_risk_monitor and RISK_MONITOR_AVAILABLE:
            strategy_id = f"{strategy.__class__.__name__}" if strategy else "unknown"
            try:
                self.risk_monitor = RiskMonitor(
                    strategy_id=strategy_id,
                    symbols=config.symbols,
                    config_path=config.alerts_config,
                    output_dir=config.risk_monitor_output
                )
                logger.info(f"[SPL-7a] RiskMonitor 已启用: {strategy_id}")
            except Exception as e:
                logger.error(f"[SPL-7a] RiskMonitor 初始化失败: {e}")
        elif config.enable_risk_monitor and not RISK_MONITOR_AVAILABLE:
            logger.warning("[SPL-7a] RiskMonitor 不可用，已跳过")

        # 运行状态
        self._running = False
        self._paused = False
        self._signal_count_today = 0
        self._last_check_time: Optional[datetime] = None
        self._daily_stats = {
            "signals_generated": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
        }

        # 回调函数
        self._on_signal_callbacks: List[Callable[[Signal], None]] = []
        self._on_order_callbacks: List[Callable[[OrderIntent], None]] = []
        self._on_fill_callbacks: List[Callable[[Trade], None]] = []
        self._on_error_callbacks: List[Callable[[Exception], None]] = []

        logger.info(f"实时交易器初始化完成，模式: {config.mode.value}")

    def register_callback(self, event_type: str, callback: Callable):
        """注册事件回调

        Args:
            event_type: 事件类型 (signal, order, fill, error)
            callback: 回调函数
        """
        if event_type == "signal":
            self._on_signal_callbacks.append(callback)
        elif event_type == "order":
            self._on_order_callbacks.append(callback)
        elif event_type == "fill":
            self._on_fill_callbacks.append(callback)
        elif event_type == "error":
            self._on_error_callbacks.append(callback)

    def start(self):
        """启动实时交易"""
        if self._running:
            logger.warning("实时交易已在运行中")
            return

        self._running = True
        logger.info("=" * 50)
        logger.info(f"实时交易启动 - 模式: {self.config.mode.value}")
        logger.info(f"交易标的: {', '.join(self.config.symbols)}")
        logger.info(f"检查间隔: {self.config.check_interval_seconds}秒")
        logger.info("=" * 50)

        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止...")
        finally:
            self.stop()

    def stop(self):
        """停止实时交易"""
        self._running = False

        # ========== SPL-7a: 关闭监控器 ==========
        if self.risk_monitor:
            try:
                self.risk_monitor.shutdown()
                monitor_stats = self.risk_monitor.get_stats()
                logger.info(f"[SPL-7a] 监控统计: {monitor_stats}")
            except Exception as e:
                logger.error(f"[SPL-7a] 关闭监控器失败: {e}")

        logger.info("实时交易已停止")

        # 打印统计
        self._print_stats()

    def pause(self):
        """暂停交易（继续接收数据但不生成信号）"""
        self._paused = True
        logger.info("交易已暂停")

    def resume(self):
        """恢复交易"""
        self._paused = False
        logger.info("交易已恢复")

    def _main_loop(self):
        """主循环"""
        while self._running:
            try:
                # 检查是否在交易时段
                if not self._is_trading_hours():
                    self._wait_for_market_open()
                    continue

                # 重置每日统计
                self._reset_daily_stats_if_needed()

                # 执行交易周期
                self._trading_cycle()

                # 等待下一次检查
                time.sleep(self.config.check_interval_seconds)

            except Exception as e:
                logger.exception("交易周期异常")
                self._notify_error(e)
                time.sleep(5)  # 出错后等待5秒再试

    def _trading_cycle(self):
        """执行一次交易周期"""
        now = datetime.now()
        self._last_check_time = now

        # 创建运行时上下文
        ctx = RunContext(
            now=now,
            trading_date=now.date(),
            mode="live",
            bar_interval=self.config.interval
        )

        for symbol in self.config.symbols:
            try:
                self._process_symbol(ctx, symbol)
            except Exception as e:
                logger.error(f"处理标的 {symbol} 时出错: {e}")
                self._notify_error(e)

    def _process_symbol(self, ctx: RunContext, symbol: str):
        """处理单个标的

        Args:
            ctx: 运行时上下文
            symbol: 标的代码
        """
        # 1. 获取最新数据
        try:
            bars = self.data_source.get_bars_until(
                symbol=symbol,
                end=ctx.now,
                interval=self.config.interval
            )

            if not bars:
                logger.warning(f"无法获取 {symbol} 的数据")
                return

            current_bar = bars[-1]

            # 记录数据获取事件
            self._log_event(
                timestamp=ctx.now,
                symbol=symbol,
                stage="data_fetch",
                payload={"bar_count": len(bars), "latest_price": float(current_bar.close)}
            )

        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return

        # ========== SPL-7a Hook 1: Pre-Decision ==========
        # 更新市场特征和基础风险指标
        if self.risk_monitor:
            try:
                self.risk_monitor.pre_decision_hook(
                    symbol=symbol,
                    bar=current_bar,
                    position=position if 'position' in locals() else None,
                    context={"mode": self.config.mode.value}
                )
            except Exception as e:
                logger.error(f"[SPL-7a] pre_decision_hook 失败: {e}")

        # 如果暂停，不生成信号
        if self._paused:
            return

        # 检查是否超过每日最大信号数
        if self._signal_count_today >= self.config.max_daily_signals:
            return

        # 2. 获取当前持仓
        position = self.broker.get_position(symbol)

        # 3. 生成信号
        if self.strategy:
            try:
                signals = self.strategy.generate_signals(bars, position)

                for signal in signals:
                    if signal.action == "HOLD":
                        continue

                    self._signal_count_today += 1
                    self._daily_stats["signals_generated"] += 1

                    # 记录信号事件
                    self._log_event(
                        timestamp=ctx.now,
                        symbol=symbol,
                        stage="signal_gen",
                        payload={
                            "action": signal.action,
                            "price": float(signal.price),
                            "quantity": signal.quantity,
                            "confidence": signal.confidence
                        },
                        reason=signal.reason
                    )

                    # 通知回调
                    self._notify_signal(signal)

                    # 4. 风控检查
                    if self.risk_manager:
                        intent = self.risk_manager.check(
                            signal=signal,
                            positions=self.broker.get_positions(),
                            cash_balance=self.broker.get_cash_balance(),
                            portfolio_value=self.broker.get_total_equity()
                        )

                        self._log_event(
                            timestamp=ctx.now,
                            symbol=symbol,
                            stage="risk_check",
                            payload={"approved": intent.approved, "reason": intent.risk_reason}
                        )

                        if not intent.approved:
                            logger.info(f"{symbol} 信号未通过风控: {intent.risk_reason}")
                            continue
                    else:
                        # 无风控，直接通过
                        from core.models import Signal
                        intent = OrderIntent(
                            signal=signal,
                            timestamp=ctx.now,
                            approved=True,
                            risk_reason="无风控，自动通过"
                        )

                    self._notify_order(intent)

                    # ========== SPL-7a Hook 2: Post-Decision ==========
                    # 记录 gating/allocator 决策结果
                    if self.risk_monitor:
                        try:
                            gating_result = {
                                "approved": intent.approved,
                                "risk_reason": intent.risk_reason
                            }
                            self.risk_monitor.post_decision_hook(
                                symbol=symbol,
                                gating_result=gating_result,
                                allocator_result=None  # TODO: 添加 allocator 结果
                            )
                        except Exception as e:
                            logger.error(f"[SPL-7a] post_decision_hook 失败: {e}")

                    # 5. 提交订单
                    if self.config.mode in [TradingMode.PAPER, TradingMode.LIVE]:
                        order = self.broker.submit_order(intent)
                        self._daily_stats["orders_submitted"] += 1

                        self._log_event(
                            timestamp=ctx.now,
                            symbol=symbol,
                            stage="order_submit",
                            payload={
                                "order_id": order.order_id if hasattr(order, 'order_id') else None,
                                "side": order.side if hasattr(order, 'side') else signal.action,
                                "quantity": order.quantity if hasattr(order, 'quantity') else signal.quantity
                            }
                        )

            except Exception as e:
                logger.error(f"策略执行失败: {e}")
                self._notify_error(e)

        # 6. 检查紧急止损
        self._check_emergency_stop_loss(position, current_bar)

        # ========== SPL-7a Hook 3: Post-Fill ==========
        # 更新 PnL/DD/spike 并触发告警判定
        if self.risk_monitor:
            try:
                self.risk_monitor.post_fill_hook(
                    symbol=symbol,
                    trade=None,  # TODO: 传入实际成交记录
                    current_price=float(current_bar.close)
                )
            except Exception as e:
                logger.error(f"[SPL-7a] post_fill_hook 失败: {e}")

    def _check_emergency_stop_loss(self, position: Optional[Position], current_bar: Bar):
        """检查紧急止损"""
        if not position or not self.config.emergency_stop_loss:
            return

        # 计算亏损比例
        if position.is_long:
            loss_ratio = (position.avg_price - current_bar.close) / position.avg_price
        else:
            loss_ratio = (current_bar.close - position.avg_price) / position.avg_price

        if loss_ratio >= self.config.emergency_stop_loss:
            logger.warning(
                f"触发紧急止损! {position.symbol} 亏损 {loss_ratio * 100:.2f}%"
            )
            # TODO: 执行市价平仓

    def _is_trading_hours(self) -> bool:
        """检查是否在交易时段"""
        # 获取市场时区的时间
        market_tz = pytz.timezone(self.config.timezone)
        now_local = datetime.now()
        now_market = now_local.astimezone(market_tz)

        # 检查是否是工作日
        if now_market.weekday() >= 5:  # 周六日
            return False

        # 检查时间（使用市场时区）
        current_time = now_market.strftime("%H:%M")

        if self.config.enable_after_hours:
            # 包含盘前盘后
            return "04:00" <= current_time <= "20:00"
        else:
            # 仅常规交易时段
            return (self.config.market_open_time <= current_time <=
                    self.config.market_close_time)

    def _wait_for_market_open(self):
        """等待市场开盘"""
        # 获取市场时区
        market_tz = pytz.timezone(self.config.timezone)
        now_local = datetime.now()
        now_market = now_local.astimezone(market_tz)

        # 计算下一个开盘时间（市场时区）
        hour, minute = map(int, self.config.market_open_time.split(":"))
        next_open = now_market.replace(
            hour=hour,
            minute=minute,
            second=0,
            microsecond=0
        )

        # 如果已经过了今天的开盘时间，等到明天
        if now_market >= next_open:
            next_open += timedelta(days=1)

        # 跳过周末
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)

        # 将市场时间转回本地时间计算等待秒数
        wait_seconds = (next_open - now_market).total_seconds()

        logger.info(f"等待市场开盘 (纽约时间 {next_open.strftime('%Y-%m-%d %H:%M')})，还有 {wait_seconds / 3600:.1f} 小时")
        time.sleep(min(wait_seconds, 300))  # 最多等5分钟再检查

    def _reset_daily_stats_if_needed(self):
        """如果需要，重置每日统计"""
        now = datetime.now()

        # 检查是否是新的一天
        if (self._last_check_time and
            self._last_check_time.date() != now.date()):
            self._signal_count_today = 0
            self._daily_stats = {
                "signals_generated": 0,
                "orders_submitted": 0,
                "orders_filled": 0,
                "orders_rejected": 0,
            }
            logger.info("新的一天，重置统计")

    def _log_event(self, timestamp: datetime, symbol: str, stage: str,
                   payload: dict, reason: str = ""):
        """记录事件"""
        if self.event_logger:
            event = Event(
                ts=timestamp,
                symbol=symbol,
                stage=stage,
                payload=payload,
                reason=reason
            )
            self.event_logger.log(event)

    def _notify_signal(self, signal: Signal):
        """通知信号回调"""
        for callback in self._on_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"信号回调错误: {e}")

    def _notify_order(self, intent: OrderIntent):
        """通知订单回调"""
        for callback in self._on_order_callbacks:
            try:
                callback(intent)
            except Exception as e:
                logger.error(f"订单回调错误: {e}")

    def _notify_fill(self, trade: Trade):
        """通知成交回调"""
        for callback in self._on_fill_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"成交回调错误: {e}")

    def _notify_error(self, error: Exception):
        """通知错误回调"""
        for callback in self._on_error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"错误回调错误: {e}")

    def _print_stats(self):
        """打印统计信息"""
        logger.info("=" * 50)
        logger.info("交易统计:")
        logger.info(f"  生成信号: {self._daily_stats['signals_generated']}")
        logger.info(f"  提交订单: {self._daily_stats['orders_submitted']}")
        logger.info(f"  成交订单: {self._daily_stats['orders_filled']}")
        logger.info(f"  拒绝订单: {self._daily_stats['orders_rejected']}")
        logger.info("=" * 50)

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            "running": self._running,
            "paused": self._paused,
            "mode": self.config.mode.value,
            "symbols": self.config.symbols,
            "signal_count_today": self._signal_count_today,
            "daily_stats": self._daily_stats,
            "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
            "portfolio_value": float(self.broker.get_total_equity()) if self.broker else 0,
            "cash_balance": float(self.broker.get_cash_balance()) if self.broker else 0,
        }


def main():
    """启动实时交易（命令行模式）"""
    import argparse

    parser = argparse.ArgumentParser(description='实时交易系统')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='交易模式')
    parser.add_argument('--symbols', nargs='+', default=['AAPL'],
                       help='交易标的')
    parser.add_argument('--interval', default='1m',
                       help='数据更新间隔')
    parser.add_argument('--cash', type=float, default=100000,
                       help='初始资金')
    parser.add_argument('--config', type=str,
                       help='配置文件路径')

    args = parser.parse_args()

    # 创建配置
    config = LiveTradingConfig(
        mode=TradingMode.PAPER if args.mode == 'paper' else TradingMode.LIVE,
        symbols=args.symbols,
        interval=args.interval
    )

    # 创建组件
    from skills.strategy.moving_average import MAStrategy
    from skills.risk.basic_risk import BasicRiskManager

    data_source = YahooFinanceSource()
    strategy = MAStrategy(fast_period=5, slow_period=20)
    risk_manager = BasicRiskManager()
    broker = PaperBroker(initial_cash=Decimal(str(args.cash)))

    # 创建实时交易器
    trader = LiveTrader(
        config=config,
        data_source=data_source,
        strategy=strategy,
        risk_manager=risk_manager,
        broker=broker
    )

    # 注册回调示例
    def on_signal(signal):
        print(f"[信号] {signal.symbol}: {signal.action} @ ${signal.price}")

    def on_order(intent):
        if intent.approved:
            print(f"[订单] {intent.signal.symbol}: 已通过风控")

    trader.register_callback("signal", on_signal)
    trader.register_callback("order", on_order)

    # 启动
    print(f"\n🚀 启动实时交易 ({args.mode} 模式)")
    print(f"   标的: {', '.join(args.symbols)}")
    print(f"   按 Ctrl+C 停止\n")

    trader.start()


if __name__ == "__main__":
    main()
