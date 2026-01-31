"""
盘后复盘报告生成

生成每日交易汇总、持仓情况、盈亏分析和信号统计
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional
import io

from core.models import Trade, Position, Signal
from broker.paper import PaperBroker


class DailyReport:
    """每日报告生成器

    生成内容：
    1. 当日交易汇总
    2. 持仓情况
    3. 盈亏分析
    4. 信号统计
    """

    def __init__(self, broker: PaperBroker, report_date: date):
        """初始化报告生成器

        Args:
            broker: 模拟经纪商
            report_date: 报告日期
        """
        self.broker = broker
        self.report_date = report_date

    def generate_text_report(self) -> str:
        """生成文本格式报告

        Returns:
            报告文本
        """
        output = io.StringIO()

        # 标题
        output.write("=" * 70 + "\n")
        output.write(f"每日交易报告 - {self.report_date}\n")
        output.write("=" * 70 + "\n\n")

        # 1. 账户概览
        self._write_account_summary(output)

        # 2. 当日交易
        self._write_daily_trades(output)

        # 3. 持仓情况
        self._write_positions(output)

        # 4. 盈亏分析
        self._write_pnl_analysis(output)

        output.write("=" * 70 + "\n")

        return output.getvalue()

    def _write_account_summary(self, output: io.StringIO):
        """写入账户概览"""
        output.write("【账户概览】\n")
        output.write("-" * 70 + "\n")

        cash = self.broker.get_cash_balance()
        equity = self.broker.get_total_equity()
        initial = self.broker.initial_cash
        pnl = self.broker.get_total_pnl()
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0

        output.write(f"初始资金: ${initial:,.2f}\n")
        output.write(f"当前现金: ${cash:,.2f}\n")
        output.write(f"总权益:   ${equity:,.2f}\n")
        output.write(f"总盈亏:   ${pnl:,.2f} ({pnl_pct:+.2f}%)\n")

        position_count = len(self.broker.get_positions())
        trade_count = len(self.broker.get_trades())

        output.write(f"持仓数量: {position_count}\n")
        output.write(f"成交次数: {trade_count}\n")
        output.write("\n")

    def _write_daily_trades(self, output: io.StringIO):
        """写入当日交易"""
        output.write("【当日交易】\n")
        output.write("-" * 70 + "\n")

        # 获取当日交易
        all_trades = self.broker.get_trades()
        daily_trades = [
            t for t in all_trades
            if t.timestamp.date() == self.report_date
        ]

        if not daily_trades:
            output.write("当日无交易\n\n")
            return

        # 按时间排序
        daily_trades.sort(key=lambda x: x.timestamp)

        for trade in daily_trades:
            time_str = trade.timestamp.strftime("%H:%M:%S")
            output.write(f"{time_str}  {trade.side:4s}  "
                        f"{trade.symbol:10s}  "
                        f"{trade.quantity:4d}股  "
                        f"@ ${trade.price:7.2f}  "
                        f"金额 ${trade.notional_value:10,.2f}  "
                        f"手续费 ${trade.commission}\n")

        # 统计
        buy_count = sum(1 for t in daily_trades if t.side == 'BUY')
        sell_count = sum(1 for t in daily_trades if t.side == 'SELL')
        buy_amount = sum(t.notional_value for t in daily_trades if t.side == 'BUY')
        sell_amount = sum(t.notional_value for t in daily_trades if t.side == 'SELL')
        commission = sum(t.commission for t in daily_trades)

        output.write(f"\n小计: 买入 {buy_count}笔 (${buy_amount:,.2f}), "
                    f"卖出 {sell_count}笔 (${sell_amount:,.2f}), "
                    f"手续费 ${commission}\n\n")

    def _write_positions(self, output: io.StringIO):
        """写入持仓情况"""
        output.write("【持仓情况】\n")
        output.write("-" * 70 + "\n")

        positions = self.broker.get_positions()

        if not positions:
            output.write("当前无持仓\n\n")
            return

        output.write(f"{'标的':<12} {'数量':>8} {'成本价':>10} {'现价':>10} "
                    f"{'市值':>12} {'浮盈':>12} {'收益率':>10}\n")

        for symbol, pos in sorted(positions.items()):
            # 简单计算现价（市值/数量）
            current_price = pos.market_value / pos.quantity if pos.quantity != 0 else 0
            pnl_pct = (pos.unrealized_pnl / pos.market_value * 100) if pos.market_value != 0 else 0

            output.write(f"{symbol:<12} {pos.quantity:>8} "
                        f"${pos.avg_price:>9.2f} "
                        f"${current_price:>9.2f} "
                        f"${pos.market_value:>11,.2f} "
                        f"${pos.unrealized_pnl:>11,.2f} "
                        f"{pnl_pct:>+9.2f}%\n")

        output.write("\n")

    def _write_pnl_analysis(self, output: io.StringIO):
        """写入盈亏分析"""
        output.write("【盈亏分析】\n")
        output.write("-" * 70 + "\n")

        # 计算已实现盈亏
        realized_pnl = Decimal("0.00")
        all_trades = self.broker.get_trades()

        # 简化的已实现盈亏计算（实际应该配对买卖）
        buy_trades = [t for t in all_trades if t.side == 'BUY']
        sell_trades = [t for t in all_trades if t.side == 'SELL']

        output.write(f"已实现盈亏: ${realized_pnl:,.2f}\n")

        # 未实现盈亏
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.broker.get_positions().values())
        output.write(f"未实现盈亏: ${unrealized_pnl:,.2f}\n")

        # 总盈亏
        total_pnl = self.broker.get_total_pnl()
        output.write(f"总盈亏:     ${total_pnl:,.2f}\n")

        # 手续费统计
        total_commission = sum(t.commission for t in all_trades)
        output.write(f"累计手续费: ${total_commission:,.2f}\n")

        output.write("\n")


class BacktestReport:
    """回测报告生成器"""

    def __init__(self, broker: PaperBroker, results: List[dict]):
        """初始化回测报告

        Args:
            broker: 模拟经纪商
            results: 每日执行结果列表
        """
        self.broker = broker
        self.results = results

    def generate_summary(self) -> str:
        """生成回测摘要报告

        Returns:
            报告文本
        """
        output = io.StringIO()

        output.write("=" * 70 + "\n")
        output.write("回测报告摘要\n")
        output.write("=" * 70 + "\n\n")

        # 基本信息
        output.write("【基本信息】\n")
        output.write("-" * 70 + "\n")

        if self.results:
            start_date = self.results[0].get('date', 'N/A')
            end_date = self.results[-1].get('date', 'N/A')
            output.write(f"回测期间: {start_date} 到 {end_date}\n")

        total_days = len(self.results)
        active_days = sum(1 for r in self.results if r.get('signals_generated', 0) > 0)
        output.write(f"交易天数: {total_days}\n")
        output.write(f"活跃天数: {active_days}\n\n")

        # 执行统计
        output.write("【执行统计】\n")
        output.write("-" * 70 + "\n")

        total_signals = sum(r.get('signals_generated', 0) for r in self.results)
        total_executed = sum(r.get('orders_filled', 0) for r in self.results)
        total_rejected = sum(r.get('orders_rejected', 0) for r in self.results)

        output.write(f"总信号数:   {total_signals}\n")
        output.write(f"执行订单:   {total_executed}\n")
        output.write(f"拒绝订单:   {total_rejected}\n")
        output.write(f"执行率:     {total_executed/total_signals*100:.2f}%\n\n")

        # 账户表现
        output.write("【账户表现】\n")
        output.write("-" * 70 + "\n")

        initial = self.broker.initial_cash
        final = self.broker.get_total_equity()
        pnl = self.broker.get_total_pnl()
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0

        output.write(f"初始资金: ${initial:,.2f}\n")
        output.write(f"最终权益: ${final:,.2f}\n")
        output.write(f"总盈亏:   ${pnl:,.2f} ({pnl_pct:+.2f}%)\n")

        # 交易统计
        trades = self.broker.get_trades()
        buy_count = sum(1 for t in trades if t.side == 'BUY')
        sell_count = sum(1 for t in trades if t.side == 'SELL')
        commission = sum(t.commission for t in trades)

        output.write(f"买入次数: {buy_count}\n")
        output.write(f"卖出次数: {sell_count}\n")
        output.write(f"总手续费: ${commission:,.2f}\n\n")

        # 持仓明细
        output.write("【当前持仓】\n")
        output.write("-" * 70 + "\n")

        positions = self.broker.get_positions()
        if positions:
            for symbol, pos in sorted(positions.items()):
                pnl_pct = (pos.unrealized_pnl / pos.market_value * 100) if pos.market_value != 0 else 0
                output.write(f"{symbol}: {pos.quantity}股, "
                           f"成本 ${pos.avg_price:.2f}, "
                           f"市值 ${pos.market_value:,.2f}, "
                           f"浮盈 ${pos.unrealized_pnl:,.2f} ({pnl_pct:+.2f}%)\n")
        else:
            output.write("无持仓\n")

        output.write("\n")
        output.write("=" * 70 + "\n")

        return output.getvalue()


if __name__ == "__main__":
    """自测代码"""
    from decimal import Decimal
    from datetime import timedelta

    print("=== DailyReport 测试 ===\n")

    # 创建模拟账户并执行一些交易
    broker = PaperBroker(initial_cash=100000)

    # 执行几笔交易
    from core.models import Signal, OrderIntent, Bar

    # 买入AAPL
    signal1 = Signal(
        symbol="AAPL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("150.00"),
        quantity=100,
        confidence=0.8,
        reason="金叉"
    )
    intent1 = OrderIntent(signal=signal1, timestamp=datetime.now(), approved=True, risk_reason="OK")
    order1 = broker.submit_order(intent1)

    # 买入GOOGL
    signal2 = Signal(
        symbol="GOOGL",
        timestamp=datetime.now(),
        action="BUY",
        price=Decimal("140.00"),
        quantity=50,
        confidence=0.8,
        reason="金叉"
    )
    intent2 = OrderIntent(signal=signal2, timestamp=datetime.now(), approved=True, risk_reason="OK")
    order2 = broker.submit_order(intent2)

    # 撮合在下一根bar
    bar = Bar(
        symbol="AAPL",
        timestamp=datetime.now() + timedelta(days=1),
        open=Decimal("151.00"),
        high=Decimal("152.00"),
        low=Decimal("149.00"),
        close=Decimal("151.50"),
        volume=100000,
        interval="1d"
    )
    broker.on_bar(bar)

    bar = Bar(
        symbol="GOOGL",
        timestamp=datetime.now() + timedelta(days=1),
        open=Decimal("141.00"),
        high=Decimal("142.00"),
        low=Decimal("139.00"),
        close=Decimal("141.50"),
        volume=100000,
        interval="1d"
    )
    broker.on_bar(bar)

    # 更新价格
    broker.update_position_prices({
        "AAPL": Decimal("155.00"),
        "GOOGL": Decimal("145.00")
    })

    # 生成报告
    report = DailyReport(broker, date.today())
    report_text = report.generate_text_report()

    print(report_text)

    print("\n✓ 报告生成测试通过")
