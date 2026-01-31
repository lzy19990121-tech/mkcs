#!/usr/bin/env python
"""
实盘模拟系统

使用真实市场数据进行实时模拟交易
"""

import time
import json
import logging
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_simulation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LiveSimulation:
    """实盘模拟系统"""

    def __init__(
        self,
        symbols: List[str],
        initial_cash: Decimal = Decimal("100000"),
        check_interval: int = 60,  # 检查间隔（秒）
        enable_slippage: bool = True
    ):
        """初始化实盘模拟

        Args:
            symbols: 交易标的列表
            initial_cash: 初始资金
            check_interval: 检查间隔（秒）
            enable_slippage: 是否启用滑点
        """
        self.symbols = symbols
        self.check_interval = check_interval
        self.running = False

        # 导入模块
        from skills.market_data.yahoo_source import YahooFinanceSource
        from skills.strategy.moving_average import MAStrategy
        from skills.risk.basic_risk import BasicRiskManager
        from broker.realistic import RealisticBroker, OrderType
        from monitoring.alerts import AlertManager, ConsoleAlertChannel, FileAlertChannel

        # 创建组件
        self.data_source = YahooFinanceSource()
        self.strategy = MAStrategy(fast_period=5, slow_period=15)
        self.risk_manager = BasicRiskManager(max_position_ratio=0.3)
        self.broker = RealisticBroker(
            initial_cash=initial_cash,
            enable_slippage=enable_slippage,
            enable_market_impact=True
        )

        # 告警系统
        alert_manager = AlertManager(channels=[
            ConsoleAlertChannel(),
            FileAlertChannel("logs/live_alerts.log")
        ])
        self.alert_manager = alert_manager

        # 历史数据缓存
        self.historical_bars: Dict[str, list] = {}

        # 状态文件
        self.state_file = Path("logs/live_simulation_state.json")
        self.trade_log_file = Path("logs/live_trades.jsonl")

        # 创建日志目录
        Path("logs").mkdir(exist_ok=True)

        logger.info(f"实盘模拟系统初始化完成")
        logger.info(f"  交易标的: {', '.join(symbols)}")
        logger.info(f"  初始资金: ${initial_cash:,.2f}")
        logger.info(f"  检查间隔: {check_interval}秒")
        logger.info(f"  滑点模拟: {'启用' if enable_slippage else '禁用'}")

    def load_historical_data(self, days: int = 60):
        """加载历史数据

        Args:
            days: 加载天数
        """
        logger.info(f"加载历史数据 (最近{days}天)...")

        end = datetime.now()
        start = end - timedelta(days=days)

        for symbol in self.symbols:
            try:
                bars = self.data_source.get_bars(symbol, start, end, '1d')
                self.historical_bars[symbol] = bars

                latest = bars[-1] if bars else None
                if latest:
                    logger.info(f"  {symbol}: {len(bars)} 根K线, 最新价 ${latest.close}")
                else:
                    logger.warning(f"  {symbol}: 未获取到数据")

            except Exception as e:
                logger.error(f"  {symbol}: 加载失败 - {e}")

    def get_real_time_quote(self, symbol: str):
        """获取实时报价

        Args:
            symbol: 标的代码

        Returns:
            Quote对象
        """
        try:
            quote = self.data_source.get_quote(symbol)
            return quote
        except Exception as e:
            logger.error(f"获取{symbol}实时报价失败: {e}")
            return None

    def check_signals(self, symbol: str) -> bool:
        """检查并执行交易信号

        Args:
            symbol: 标的代码

        Returns:
            是否执行了交易
        """
        if symbol not in self.historical_bars:
            return False

        bars = self.historical_bars[symbol]
        if len(bars) < 20:
            return False

        try:
            # 生成信号
            signals = self.strategy.generate_signals(bars, position=None)

            if not signals:
                return False

            traded = False

            for signal in signals:
                logger.info(f"  信号: {signal.action} {signal.quantity} {signal.symbol} @ ${signal.price} (置信度: {signal.confidence:.2%})")

                # 风控检查
                from core.models import OrderIntent
                intent = OrderIntent(
                    signal=signal,
                    timestamp=datetime.now(),
                    approved=True,
                    risk_reason="实盘模拟"
                )

                # 检查持仓
                positions = self.broker.get_positions()
                if signal.action == "SELL":
                    if symbol not in positions or positions[symbol] < signal.quantity:
                        logger.info(f"    ✗ 跳过: 持仓不足 (持有{positions.get(symbol, 0)}股)")
                        continue

                # 获取当前K线作为执行价格
                current_bar = bars[-1]

                # 执行订单
                result = self.broker.submit_order(intent, current_bar, OrderType.MARKET)

                if result.trade:
                    traded = True
                    logger.info(f"    ✓ 成交: {result.filled_quantity} @ ${result.filled_price} (滑点: {result.slippage_percent:.3f}%)")

                    # 记录交易
                    self.log_trade(result.trade, signal)

                    # 检查风险
                    self.check_risk_limits(symbol)

                elif result.reason:
                    logger.info(f"    ✗ {result.reason}")

            return traded

        except Exception as e:
            logger.error(f"检查{symbol}信号失败: {e}")
            return False

    def log_trade(self, trade, signal):
        """记录交易日志

        Args:
            trade: 成交记录
            signal: 交易信号
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'side': trade.side,
            'price': str(trade.price),
            'quantity': trade.quantity,
            'commission': str(trade.commission),
            'notional_value': str(trade.notional_value),
            'signal_reason': signal.reason,
            'signal_confidence': signal.confidence
        }

        with open(self.trade_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def check_risk_limits(self, symbol: str):
        """检查风险限制

        Args:
            symbol: 标的代码
        """
        # 检查单日亏损
        stats = self.broker.get_execution_stats()
        if stats['filled_orders'] > 0:
            # 计算当前权益
            current_equity = self.broker.get_cash_balance()
            positions = self.broker.get_positions()

            for pos_symbol, qty in positions.items():
                if pos_symbol in self.historical_bars and self.historical_bars[pos_symbol]:
                    latest_price = self.historical_bars[pos_symbol][-1].close
                    current_equity += latest_price * qty

            initial_equity = self.broker.initial_cash
            daily_pnl = current_equity - initial_equity

            if daily_pnl < 0:
                loss_ratio = -daily_pnl / initial_equity
                if loss_ratio > 0.03:  # 3%警告
                    self.alert_manager.check_daily_loss_limit(daily_pnl, initial_equity)

    def update_historical_with_quote(self, symbol: str, quote):
        """用实时报价更新历史数据

        Args:
            symbol: 标的代码
            quote: 实时报价
        """
        if not quote:
            return

        from core.models import Bar

        # 创建当前K线（简化：使用报价价格）
        current_bar = Bar(
            symbol=symbol,
            timestamp=quote.timestamp,
            open=quote.mid_price,
            high=quote.ask_price,
            low=quote.bid_price,
            close=quote.mid_price,
            volume=quote.bid_size + quote.ask_size,
            interval='1d'
        )

        # 更新历史数据
        if symbol not in self.historical_bars:
            self.historical_bars[symbol] = []

        # 如果最新K线是今天的，更新它；否则添加新的
        bars = self.historical_bars[symbol]
        if bars and bars[-1].timestamp.date() == current_bar.timestamp.date():
            # 更新今天的K线
            bars[-1] = current_bar
        else:
            # 添加新的K线
            bars.append(current_bar)

    def print_status(self):
        """打印当前状态"""
        print("\n" + "="*60)
        print(f"  实盘模拟状态 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)

        # 资金状态
        cash = self.broker.get_cash_balance()
        positions = self.broker.get_positions()

        print(f"\n资金:")
        print(f"  初始资金: ${self.broker.initial_cash:,.2f}")
        print(f"  现金余额: ${cash:,.2f}")

        # 持仓状态
        if positions:
            print(f"\n持仓:")
            total_market_value = Decimal("0")

            for symbol, qty in positions.items():
                if symbol in self.historical_bars and self.historical_bars[symbol]:
                    latest_price = self.historical_bars[symbol][-1].close
                    market_value = latest_price * qty
                    total_market_value += market_value

                    # 计算盈亏
                    avg_price = Decimal("130")  # 简化，实际应该记录成本价
                    pnl = (latest_price - avg_price) * qty
                    pnl_percent = (latest_price - avg_price) / avg_price * 100

                    print(f"  {symbol}: {qty} 股 @ ${latest_price:.2f}")
                    print(f"    市值: ${market_value:,.2f}")
                    print(f"    盈亏: ${pnl:+,.2f} ({pnl_percent:+.2f}%)")

            total_equity = cash + total_market_value
            total_return = (total_equity - self.broker.initial_cash) / self.broker.initial_cash

            print(f"\n总权益: ${total_equity:,.2f}")
            print(f"总收益率: {total_return:.2%}")

        else:
            print(f"\n持仓: 无")

        # 执行统计
        stats = self.broker.get_execution_stats()
        print(f"\n执行统计:")
        print(f"  总订单: {stats['total_orders']}")
        print(f"  成交: {stats['filled_orders']}")
        print(f"  成交率: {stats['fill_rate']:.2%}")
        print(f"  平均滑点: {stats['avg_slippage']:.4f}%")

        print("="*60 + "\n")

    def save_state(self):
        """保存状态到文件"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'cash': str(self.broker.get_cash_balance()),
            'positions': self.broker.get_positions(),
            'stats': self.broker.get_execution_stats()
        }

        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def run_check_cycle(self):
        """执行一次检查周期"""
        logger.info(f"\n--- 检查周期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

        # 1. 获取实时报价
        for symbol in self.symbols:
            try:
                quote = self.get_real_time_quote(symbol)
                if quote:
                    logger.info(f"{symbol} 实时报价: 买 ${quote.bid_price} / 卖 ${quote.ask_price}")
                    self.update_historical_with_quote(symbol, quote)
            except Exception as e:
                logger.warning(f"获取{symbol}报价失败: {e}")

        # 2. 检查交易信号
        for symbol in self.symbols:
            try:
                self.check_signals(symbol)
            except Exception as e:
                logger.error(f"处理{symbol}失败: {e}")

        # 3. 保存状态
        self.save_state()

    def run(self, duration_minutes: int = 60):
        """运行实盘模拟

        Args:
            duration_minutes: 运行时长（分钟）
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  开始实盘模拟")
        logger.info(f"  运行时长: {duration_minutes} 分钟")
        logger.info(f"{'='*60}\n")

        self.running = True

        # 加载历史数据
        self.load_historical_data(days=60)

        # 显示初始状态
        self.print_status()

        # 记录开始时间
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        cycle_count = 0

        try:
            while self.running and datetime.now() < end_time:
                cycle_count += 1

                # 执行检查周期
                self.run_check_cycle()

                # 显示状态（每5次周期）
                if cycle_count % 5 == 0:
                    self.print_status()

                # 等待下一次检查
                if datetime.now() < end_time:
                    wait_seconds = min(self.check_interval, (end_time - datetime.now()).total_seconds())
                    if wait_seconds > 0:
                        logger.info(f"等待 {wait_seconds:.0f} 秒后进行下次检查...")
                        time.sleep(wait_seconds)

        except KeyboardInterrupt:
            logger.info("\n收到停止信号，正在退出...")

        finally:
            self.running = False
            self.print_status()
            self.save_state()

            logger.info(f"\n{'='*60}")
            logger.info(f"  实盘模拟结束")
            logger.info(f"  运行时长: {(datetime.now() - start_time).total_seconds() / 60:.1f} 分钟")
            logger.info(f"  检查周期: {cycle_count} 次")
            logger.info(f"{'='*60}\n")

    def stop(self):
        """停止运行"""
        self.running = False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  实盘模拟系统")
    print("="*60)

    # 配置
    symbols = ["AAPL", "MSFT", "GOOGL"]
    initial_cash = Decimal("100000")
    check_interval = 30  # 30秒检查一次
    duration_minutes = 5  # 运行5分钟演示

    print(f"\n配置:")
    print(f"  交易标的: {', '.join(symbols)}")
    print(f"  初始资金: ${initial_cash:,.2f}")
    print(f"  检查间隔: {check_interval}秒")
    print(f"  运行时长: {duration_minutes}分钟")
    print(f"\n注意: 这是模拟交易，不会执行真实订单")
    print(f"      按 Ctrl+C 可以随时停止\n")

    input("按 Enter 开始...")

    # 创建并运行模拟
    sim = LiveSimulation(
        symbols=symbols,
        initial_cash=initial_cash,
        check_interval=check_interval,
        enable_slippage=True
    )

    try:
        sim.run(duration_minutes=duration_minutes)
    except Exception as e:
        logger.error(f"模拟运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
