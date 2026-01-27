"""
交易Agent运行器

负责任务编排和状态管理，协调各个skill完成回测
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Optional, Dict
import sys
import argparse

from core.models import Bar, Position, OrderIntent, Signal
from skills.market_data.mock_source import MarketDataSource, MockMarketSource
from skills.strategy.base import Strategy
from skills.strategy.moving_average import MAStrategy
from skills.risk.basic_risk import RiskManager, BasicRiskManager
from skills.session.trading_session import TradingSession
from broker.paper import PaperBroker
from storage.db import TradeDB


class TradingAgent:
    """自动交易Agent

    职责：
    1. 编排各个skill
    2. 执行回测主循环
    3. 记录交易数据
    """

    def __init__(
        self,
        data_source: MarketDataSource,
        strategy: Strategy,
        risk_manager: RiskManager,
        broker: PaperBroker,
        db: Optional[TradeDB] = None
    ):
        """初始化Agent

        Args:
            data_source: 市场数据源
            strategy: 交易策略
            risk_manager: 风控管理器
            broker: 模拟经纪商
            db: 数据库（可选）
        """
        self.data_source = data_source
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.broker = broker
        self.db = db

        # 初始化风控管理器的资金
        self.risk_manager.set_capital(float(self.broker.initial_cash))

    def run_one_day(self, trading_date: date, symbols: List[str]) -> dict:
        """运行单个交易日

        Args:
            trading_date: 交易日期
            symbols: 需要交易的标的列表

        Returns:
            当日执行结果统计
        """
        result = {
            "date": trading_date,
            "symbols_processed": 0,
            "signals_generated": 0,
            "orders_executed": 0,
            "orders_rejected": 0
        }

        # 检查是否为交易日
        market = TradingSession.get_market_type(symbols[0]) if symbols else 'US'
        if not TradingSession.is_trading_day(trading_date, market):
            return result

        # 获取当日市场时段
        sessions = TradingSession.get_trading_sessions(trading_date, market)
        if not sessions:
            return result

        # 为每个标的执行交易逻辑
        for symbol in symbols:
            result["symbols_processed"] += 1

            # 1. 获取历史数据
            start_date = trading_date - timedelta(days=self.strategy.get_min_bars_required() + 10)
            end_time = datetime.combine(trading_date, sessions[0][0].time())

            bars = self.data_source.get_bars(
                symbol,
                datetime.combine(start_date, datetime.min.time()),
                end_time,
                "1d"
            )

            if len(bars) < self.strategy.get_min_bars_required():
                continue

            # 2. 获取当前持仓
            position = self.broker.get_position(symbol)

            # 3. 策略生成信号
            signals = self.strategy.generate_signals(bars, position)

            for signal in signals:
                result["signals_generated"] += 1

                # 4. 风控检查
                intent = self.risk_manager.check(
                    signal,
                    self.broker.get_positions(),
                    float(self.broker.get_cash_balance())
                )

                # 5. 执行交易
                if intent.can_execute:
                    trade = self.broker.execute_order(intent)
                    if trade:
                        result["orders_executed"] += 1
                        if self.db:
                            self.db.save_trade(trade)
                    else:
                        result["orders_rejected"] += 1
                else:
                    result["orders_rejected"] += 1

        # 6. 更新持仓价格（使用收盘价）
        closing_prices = {}
        for symbol in symbols:
            try:
                bars = self.data_source.get_bars(
                    symbol,
                    datetime.combine(trading_date, datetime.min.time()),
                    datetime.combine(trading_date, datetime.max.time()),
                    "1d"
                )
                if bars:
                    closing_prices[symbol] = bars[-1].close
            except:
                pass

        if closing_prices:
            self.broker.update_position_prices(closing_prices)

        # 7. 保存持仓快照
        if self.db:
            for position in self.broker.get_positions().values():
                self.db.save_position(position, trading_date)

        # 更新当日盈亏
        daily_pnl = self.broker.get_total_pnl()
        self.risk_manager.update_daily_pnl(daily_pnl)

        return result

    def run_backtest(
        self,
        start: date,
        end: date,
        symbols: List[str],
        verbose: bool = True
    ) -> List[dict]:
        """运行回测

        Args:
            start: 开始日期
            end: 结束日期
            symbols: 交易标的列表
            verbose: 是否打印详细信息

        Returns:
            每日执行结果列表
        """
        results = []
        market = TradingSession.get_market_type(symbols[0]) if symbols else 'US'

        # 获取所有交易日
        trading_days = TradingSession.get_trading_days(start, end, market)

        if verbose:
            print(f"\n开始回测: {start} 到 {end}")
            print(f"交易标的: {', '.join(symbols)}")
            print(f"交易日数量: {len(trading_days)}")
            print(f"初始资金: ${self.broker.get_cash_balance():,.2f}\n")

        # 逐日运行
        for i, trading_date in enumerate(trading_days, 1):
            daily_result = self.run_one_day(trading_date, symbols)
            results.append(daily_result)

            if verbose and daily_result["signals_generated"] > 0:
                print(f"[{i}/{len(trading_days)}] {trading_date}: "
                      f"信号={daily_result['signals_generated']}, "
                      f"成交={daily_result['orders_executed']}, "
                      f"拒绝={daily_result['orders_rejected']}")

        if verbose:
            self._print_summary()

        return results

    def _print_summary(self):
        """打印回测摘要"""
        print("\n" + "="*60)
        print("回测摘要")
        print("="*60)

        print(f"\n账户状态:")
        print(f"  最终现金: ${self.broker.get_cash_balance():,.2f}")
        print(f"  总权益: ${self.broker.get_total_equity():,.2f}")
        print(f"  总盈亏: ${self.broker.get_total_pnl():,.2f} ({self.broker.get_total_pnl()/self.broker.initial_cash*100:.2f}%)")

        print(f"\n持仓情况:")
        positions = self.broker.get_positions()
        if positions:
            for symbol, pos in positions.items():
                print(f"  {symbol}: {pos.quantity}股, 成本 ${pos.avg_price}, "
                      f"市值 ${pos.market_value}, "
                      f"浮盈 ${pos.unrealized_pnl} ({pos.unrealized_pnl/pos.market_value*100:.2f}%)")
        else:
            print("  无持仓")

        print(f"\n交易统计:")
        trades = self.broker.get_trades()
        print(f"  总成交次数: {len(trades)}")
        buy_count = sum(1 for t in trades if t.side == 'BUY')
        sell_count = sum(1 for t in trades if t.side == 'SELL')
        print(f"  买入次数: {buy_count}")
        print(f"  卖出次数: {sell_count}")

        total_commission = sum(t.commission for t in trades)
        print(f"  总手续费: ${total_commission}")

        print("="*60)


def create_default_agent(initial_cash: float = 1000000, db_path: Optional[str] = None) -> TradingAgent:
    """创建默认配置的Agent

    Args:
        initial_cash: 初始资金
        db_path: 数据库路径（可选）

    Returns:
        配置好的TradingAgent
    """
    # 创建各个组件
    data_source = MockMarketSource(seed=42)
    strategy = MAStrategy(fast_period=5, slow_period=20)
    risk_manager = BasicRiskManager(
        max_position_ratio=0.2,
        max_positions=5,
        blacklist=set(),
        max_daily_loss_ratio=0.05
    )
    broker = PaperBroker(initial_cash=initial_cash, commission_per_share=0.01)

    # 创建数据库（可选）
    db = TradeDB(db_path) if db_path else None

    # 创建Agent
    agent = TradingAgent(data_source, strategy, risk_manager, broker, db)

    return agent


if __name__ == "__main__":
    """自测代码"""
    parser = argparse.ArgumentParser(description='运行自动交易回测')
    parser.add_argument('--start', type=str, default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--cash', type=float, default=100000, help='初始资金')
    parser.add_argument('--db', type=str, default=None, help='数据库路径')
    parser.add_argument('--quiet', action='store_true', help='静默模式')

    args = parser.parse_args()

    # 解析日期
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    # 创建Agent
    agent = create_default_agent(initial_cash=args.cash, db_path=args.db)

    # 运行回测
    symbols = ["AAPL", "GOOGL", "MSFT"]

    try:
        results = agent.run_backtest(
            start=start_date,
            end=end_date,
            symbols=symbols,
            verbose=not args.quiet
        )

        if not args.quiet:
            print(f"\n✓ 回测完成")

    except Exception as e:
        print(f"\n✗ 回测出错: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
