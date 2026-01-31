"""
日复盘生成器

从事件日志生成每日交易复盘报告
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from collections import defaultdict

from events.event_log import EventLogger, get_event_logger


class DailyReviewer:
    """日复盘生成器

    生成内容：
    1. 交易摘要
    2. 执行回顾
    3. 盈亏分析
    4. 信号质量
    5. 问题与改进
    """

    def __init__(self, event_logger: Optional[EventLogger] = None):
        """初始化日复盘生成器

        Args:
            event_logger: 事件日志器（可选）
        """
        self.event_logger = event_logger or get_event_logger()

    def generate_review(
        self,
        target_date: date,
        broker=None
    ) -> str:
        """生成日复盘

        Args:
            target_date: 目标日期
            broker: 模拟经纪商（可选，用于获取账户信息）

        Returns:
            复盘报告（Markdown格式）
        """
        lines = []

        # 标题
        lines.append(f"# 交易复盘 - {target_date.strftime('%Y年%m月%d日')}\n")
        lines.append("---\n")

        # 1. 交易摘要
        lines.append("## 1. 交易摘要\n")
        lines.extend(self._generate_summary(target_date))
        lines.append("\n")

        # 2. 执行回顾
        lines.append("## 2. 执行回顾\n")
        lines.extend(self._generate_execution_review(target_date))
        lines.append("\n")

        # 3. 信号质量
        lines.append("## 3. 信号质量\n")
        lines.extend(self._generate_signal_quality(target_date))
        lines.append("\n")

        # 4. 盈亏分析
        lines.append("## 4. 盈亏分析\n")
        if broker:
            lines.extend(self._generate_pnl_analysis(broker))
        else:
            lines.append("未提供账户信息\n")
        lines.append("\n")

        # 5. 时间线
        lines.append("## 5. 时间线\n")
        lines.extend(self._generate_timeline(target_date))
        lines.append("\n")

        # 6. 问题与改进
        lines.append("## 6. 问题与改进\n")
        lines.extend(self._generate_improvements(target_date))
        lines.append("\n")

        return ''.join(lines)

    def _generate_summary(self, target_date: date) -> List[str]:
        """生成交易摘要"""
        lines = []

        # 查询当日事件
        events = self.event_logger.query_events(
            start=datetime.combine(target_date, datetime.min.time()),
            end=datetime.combine(target_date, datetime.max.time())
        )

        # 统计
        signal_count = len([e for e in events if e.get('stage') == 'signal_gen'])
        risk_pass = len([e for e in events if e.get('stage') == 'risk_check' and e['payload'].get('approved')])
        risk_fail = len([e for e in events if e.get('stage') == 'risk_check' and not e['payload'].get('approved')])
        trade_count = len([e for e in events if e.get('stage') == 'order_fill'])

        lines.append(f"- **信号数量**: {signal_count}\n")
        lines.append(f"- **风控通过**: {risk_pass}\n")
        lines.append(f"- **风控拒绝**: {risk_fail}\n")
        lines.append(f"- **成交数量**: {trade_count}\n")
        lines.append(f"- **执行率**: {trade_count/signal_count*100:.1f}%\n" if signal_count > 0 else "- **执行率**: N/A\n")

        # 按标的统计
        symbols = [str(e.get('symbol')) for e in events if e.get('symbol') is not None]
        unique_symbols = sorted(set(symbols))
        lines.append(f"- **涉及标的**: {len(unique_symbols)} 只\n")
        lines.append(f"- **标的列表**: {', '.join(unique_symbols)}\n")

        return lines

    def _generate_execution_review(self, target_date: date) -> List[str]:
        """生成执行回顾"""
        lines = []

        # 查询订单事件
        fill_events = self.event_logger.query_events(
            start=datetime.combine(target_date, datetime.min.time()),
            end=datetime.combine(target_date, datetime.max.time()),
            stage='order_fill'
        )

        submit_events = self.event_logger.query_events(
            start=datetime.combine(target_date, datetime.min.time()),
            end=datetime.combine(target_date, datetime.max.time()),
            stage='order_submit'
        )

        reject_events = self.event_logger.query_events(
            start=datetime.combine(target_date, datetime.min.time()),
            end=datetime.combine(target_date, datetime.max.time()),
            stage='order_reject'
        )

        lines.append(f"- 提交订单: {len(submit_events)}\n")
        lines.append(f"- 成交订单: {len(fill_events)}\n")
        lines.append(f"- 拒绝订单: {len(reject_events)}\n\n")

        if not fill_events:
            lines.append("当日无成交\n")
        else:
            lines.append("### 成交明细\n\n")
            lines.append("| 时间 | 标的 | 方向 | 价格 | 数量 | 原因 |\n")
            lines.append("|------|------|------|------|------|------|\n")

            for event in fill_events:
                ts = datetime.fromisoformat(event['ts']).strftime('%H:%M:%S')
                symbol = event['symbol']
                payload = event['payload']
                side = payload.get('side', 'N/A')
                price = payload.get('price', 0)
                quantity = payload.get('quantity', 0)
                reason = event['reason']

                lines.append(f"| {ts} | {symbol} | {side} | ${price:.2f} | {quantity} | {reason} |\n")

        if reject_events:
            lines.append("\n### 拒绝明细\n\n")
            lines.append("| 时间 | 标的 | 方向 | 数量 | 原因 |\n")
            lines.append("|------|------|------|------|------|\n")

            for event in reject_events:
                ts = datetime.fromisoformat(event['ts']).strftime('%H:%M:%S')
                symbol = event['symbol']
                payload = event['payload']
                side = payload.get('side', payload.get('action', 'N/A'))
                quantity = payload.get('quantity', 0)
                reason = event['reason']

                lines.append(f"| {ts} | {symbol} | {side} | {quantity} | {reason} |\n")
        return lines

    def _generate_signal_quality(self, target_date: date) -> List[str]:
        """生成信号质量分析"""
        lines = []

        # 查询信号生成事件
        events = self.event_logger.query_events(
            start=datetime.combine(target_date, datetime.min.time()),
            end=datetime.combine(target_date, datetime.max.time()),
            stage='signal_gen'
        )

        if not events:
            lines.append("当日无信号\n")
            return lines

        lines.append("### 信号统计\n\n")

        # 按类型分组
        buy_signals = [e for e in events if e['payload'].get('action') == 'BUY']
        sell_signals = [e for e in events if e['payload'].get('action') == 'SELL']
        hold_signals = [e for e in events if e['payload'].get('action') == 'HOLD']

        lines.append(f"- **买入信号**: {len(buy_signals)}\n")
        lines.append(f"- **卖出信号**: {len(sell_signals)}\n")
        lines.append(f"- **观望信号**: {len(hold_signals)}\n")

        # 平均置信度
        all_conf = [e['payload'].get('confidence', 0) for e in events]
        if all_conf:
            avg_conf = sum(all_conf) / len(all_conf)
            lines.append(f"- **平均置信度**: {avg_conf:.2%}\n")

        # 高置信度信号
        high_conf = [e for e in events if e['payload'].get('confidence', 0) >= 0.8]
        lines.append(f"- **高置信度(≥80%)**: {len(high_conf)}\n")

        lines.append("\n### 信号详情\n\n")
        lines.append("| 标的 | 方向 | 置信度 | 原因 |\n")
        lines.append("|------|------|--------|------|\n")

        for event in events:
            symbol = event['symbol']
            action = event['payload'].get('action', 'N/A')
            confidence = event['payload'].get('confidence', 0)
            reason = event['reason']

            lines.append(f"| {symbol} | {action} | {confidence:.2%} | {reason} |\n")

        return lines

    def _generate_pnl_analysis(self, broker) -> List[str]:
        """生成盈亏分析"""
        lines = []

        # 账户概况
        initial = broker.initial_cash
        final = broker.get_total_equity()
        pnl = broker.get_total_pnl()
        pnl_pct = (pnl / initial * 100) if initial > 0 else 0

        lines.append(f"- **初始资金**: ${initial:,.2f}\n")
        lines.append(f"- **最终权益**: ${final:,.2f}\n")
        lines.append(f"- **总盈亏**: ${pnl:,.2f} ({pnl_pct:+.2f}%)\n")

        # 持仓情况
        positions = broker.get_positions()
        if positions:
            lines.append(f"\n### 当前持仓\n\n")
            lines.append("| 标的 | 数量 | 成本 | 市值 | 浮盈 |\n")
            lines.append("|------|------|------|------|------|\n")

            for symbol, pos in positions.items():
                lines.append(f"| {symbol} | {pos.quantity} | ${pos.avg_price:.2f} | "
                           f"${pos.market_value:,.2f} | ${pos.unrealized_pnl:,.2f} |\n")
        else:
            lines.append(f"\n当前无持仓\n")

        return lines

    def _generate_timeline(self, target_date: date) -> List[str]:
        """生成时间线"""
        lines = []

        # 查询所有事件
        events = self.event_logger.query_events(
            start=datetime.combine(target_date, datetime.min.time()),
            end=datetime.combine(target_date, datetime.max.time())
        )

        if not events:
            lines.append("当日无事件记录\n")
            return lines

        lines.append("### 事件时间线\n\n")

        # 按时间排序
        events.sort(key=lambda x: x['ts'])

        # 按小时分组
        by_hour = defaultdict(list)
        for event in events:
            dt = datetime.fromisoformat(event['ts'])
            hour = dt.hour
            by_hour[hour].append(event)

        for hour in sorted(by_hour.keys()):
            lines.append(f"#### {hour:02d}:00 - {hour:02d}:59\n\n")

            for event in by_hour[hour]:
                dt = datetime.fromisoformat(event['ts'])
                time_str = dt.strftime('%H:%M:%S')
                stage = event['stage']
                symbol = event.get('symbol', 'SYSTEM')
                reason = event['reason']

                lines.append(f"- **{time_str}** [{symbol}] {stage}: {reason}\n")

            lines.append("\n")

        return lines

    def _generate_improvements(self, target_date: date) -> List[str]:
        """生成问题与改进"""
        lines = []

        # 查询风控拒绝的事件
        rejected = self.event_logger.query_events(
            start=datetime.combine(target_date, datetime.min.time()),
            end=datetime.combine(target_date, datetime.max.time()),
            stage='risk_check'
        )

        rejected = [e for e in rejected if not e['payload'].get('approved', True)]

        if rejected:
            lines.append("### 风控拒绝分析\n\n")
            for event in rejected:
                symbol = event['symbol']
                reason = event['reason']
                lines.append(f"- **{symbol}**: {reason}\n")
            lines.append("\n")

        # TODO：添加更多分析逻辑
        lines.append("### 待改进项\n\n")
        lines.append("- 数据源接入真实市场数据\n")
        lines.append("- 增加更多策略类型\n")
        lines.append("- 优化风控规则\n")

        return lines


if __name__ == "__main__":
    """自测代码"""
    from events.event_log import EventLogger, Event
    from broker.paper import PaperBroker
    from datetime import datetime
    from decimal import Decimal

    print("=== DailyReviewer 测试 ===\n")

    # 创建临时日志器并添加测试数据
    logger = EventLogger(log_dir="test_logs")

    target_date = datetime.now().date()

    # 添加测试事件
    test_events = [
        Event(
            ts=datetime.combine(target_date, datetime.min.time()).replace(hour=9, minute=30),
            symbol="AAPL",
            stage="data_fetch",
            payload={"bars_count": 21},
            reason="获取K线数据"
        ),
        Event(
            ts=datetime.combine(target_date, datetime.min.time()).replace(hour=9, minute=35),
            symbol="AAPL",
            stage="signal_gen",
            payload={"action": "BUY", "confidence": 0.85},
            reason="金叉买入"
        ),
        Event(
            ts=datetime.combine(target_date, datetime.min.time()).replace(hour=9, minute=36),
            symbol="AAPL",
            stage="risk_check",
            payload={"approved": True},
            reason="通过风控"
        ),
        Event(
            ts=datetime.combine(target_date, datetime.min.time()).replace(hour=9, minute=37),
            symbol="AAPL",
            stage="order_fill",
            payload={"side": "BUY", "price": 150.0, "quantity": 100},
            reason="订单成交"
        ),
        Event(
            ts=datetime.combine(target_date, datetime.min.time()).replace(hour=10, minute=0),
            symbol="GOOGL",
            stage="signal_gen",
            payload={"action": "SELL", "confidence": 0.75},
            reason="死叉卖出"
        ),
        Event(
            ts=datetime.combine(target_date, datetime.min.time()).replace(hour=10, minute=1),
            symbol="GOOGL",
            stage="risk_check",
            payload={"approved": False},
            reason="持仓不足"
        ),
    ]

    for event in test_events:
        logger.log(event)

    # 创建模拟账户
    broker = PaperBroker(initial_cash=100000)

    # 生成复盘
    reviewer = DailyReviewer(event_logger=logger)
    review = reviewer.generate_review(target_date, broker)

    print(review)

    # 清理
    logger.close()

    import shutil
    import os
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")

    print("✓ 测试完成")
