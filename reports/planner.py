"""
日计划生成器

从事件日志生成每日交易计划
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from pathlib import Path

from events.event_log import EventLogger, get_event_logger


class DailyPlanner:
    """日计划生成器

    生成内容：
    1. 市场概况
    2. 今日重点关注
    3. 交易计划
    4. 风控要点
    5. 其他事项
    """

    def __init__(self, event_logger: Optional[EventLogger] = None):
        """初始化日计划生成器

        Args:
            event_logger: 事件日志器（可选，默认使用全局实例）
        """
        self.event_logger = event_logger or get_event_logger()

    def generate_plan(
        self,
        target_date: date,
        symbols: List[str],
        market: str = 'US'
    ) -> str:
        """生成日计划

        Args:
            target_date: 目标日期
            symbols: 关注标的列表
            market: 市场类型

        Returns:
            计划文档（Markdown格式）
        """
        lines = []

        # 标题
        lines.append(f"# 交易计划 - {target_date.strftime('%Y年%m月%d日')}\n")
        lines.append(f"**市场**: {market}  **星期**: {self._get_weekday(target_date)}\n")
        lines.append("---\n")

        # 1. 市场概况
        lines.append("## 1. 市场概况\n")
        lines.extend(self._generate_market_overview(target_date, market))
        lines.append("\n")

        # 2. 今日关注标的
        lines.append("## 2. 关注标的\n")
        lines.extend(self._generate_watchlist(symbols))
        lines.append("\n")

        # 3. 交易计划
        lines.append("## 3. 交易计划\n")
        lines.extend(self._generate_trading_plan(target_date, symbols))
        lines.append("\n")

        # 4. 风控要点
        lines.append("## 4. 风控要点\n")
        lines.extend(self._generate_risk_notes())
        lines.append("\n")

        # 5. 历史事件参考
        lines.append("## 5. 历史参考\n")
        lines.extend(self._generate_historical_notes(target_date, symbols))
        lines.append("\n")

        return ''.join(lines)

    def _get_weekday(self, dt: date) -> str:
        """获取星期几"""
        weekdays = ['一', '二', '三', '四', '五', '六', '日']
        return f"星期{weekdays[dt.weekday()]}"

    def _generate_market_overview(
        self,
        target_date: date,
        market: str
    ) -> List[str]:
        """生成市场概况"""
        lines = []

        # 检查是否为交易日
        from skills.session.trading_session import TradingSession

        is_trading = TradingSession.is_trading_day(target_date, market)

        lines.append(f"- **交易状态**: {'✅ 交易日' if is_trading else '❌ 非交易日'}\n")

        if is_trading:
            sessions = TradingSession.get_trading_sessions(target_date, market)
            lines.append(f"- **交易时段**: ")
            for i, (start, end) in enumerate(sessions, 1):
                lines.append(f"{start.strftime('%H:%M')}-{end.strftime('%H:%M')}")
                if i < len(sessions):
                    lines.append(", ")
            lines.append("\n")

        # 经济日历（TODO：接入真实数据）
        lines.append(f"- **重要事件**: 无特别安排\n")

        return lines

    def _generate_watchlist(self, symbols: List[str]) -> List[str]:
        """生成关注标的列表"""
        lines = []

        lines.append("| 标的 | 昨收 | 关注理由 | 策略 |\n")
        lines.append("|------|------|----------|------|\n")

        for symbol in symbols:
            # TODO：从数据库或API获取实时数据
            lines.append(f"| {symbol} | - | 策略关注 | MA交叉 |\n")

        return lines

    def _generate_trading_plan(
        self,
        target_date: date,
        symbols: List[str]
    ) -> List[str]:
        """生成交易计划"""
        lines = []

        lines.append("### 预期操作\n\n")

        for symbol in symbols:
            # 查询历史事件
            yesterday = target_date - timedelta(days=1)
            events = self.event_logger.query_events(
                start=datetime.combine(yesterday, datetime.min.time()),
                end=datetime.combine(target_date, datetime.max.time()),
                symbol=symbol
            )

            # 分析最近信号
            latest_signals = [e for e in events if e.get('stage') == 'signal_gen']

            if latest_signals:
                latest = latest_signals[-1]
                action = latest['payload'].get('action', 'HOLD')
                confidence = latest['payload'].get('confidence', 0)
                lines.append(f"**{symbol}**:\n")
                lines.append(f"- 最新信号: {action} (置信度: {confidence:.2%})\n")
                lines.append(f"- 原因: {latest['reason']}\n")
            else:
                lines.append(f"**{symbol}**:\n")
                lines.append(f"- 等待信号生成\n")

            lines.append("\n")

        return lines

    def _generate_risk_notes(self) -> List[str]:
        """生成风控要点"""
        lines = []

        lines.append("- **单只股票仓位限制**: 20%\n")
        lines.append("- **持仓数量限制**: 5只\n")
        lines.append("- **单日亏损限制**: 5%\n")
        lines.append("- **黑名单**: 无\n")
        lines.append("\n")
        lines.append("**今日风控重点**:\n")
        lines.append("- 监控总仓位不超过80%\n")
        lines.append("- 注意开盘和收盘波动\n")
        lines.append("- 及时止损，单笔亏损不超过2%\n")

        return lines

    def _generate_historical_notes(
        self,
        target_date: date,
        symbols: List[str]
    ) -> List[str]:
        """生成历史参考"""
        lines = []

        # 查询最近7天的同一星期几的历史
        for i in range(1, 4):
            ref_date = target_date - timedelta(weeks=i)
            if not ref_date >= date(2024, 1, 1):
                continue

            events = self.event_logger.query_events(
                start=datetime.combine(ref_date, datetime.min.time()),
                end=datetime.combine(ref_date, datetime.max.time())
            )

            if events:
                lines.append(f"### {ref_date.strftime('%Y-%m-%d')} ({self._get_weekday(ref_date)})\n\n")

                # 统计
                signal_count = len([e for e in events if e.get('stage') == 'signal_gen'])
                trade_count = len([e for e in events if e.get('stage') == 'order_exec'])

                lines.append(f"- 信号数: {signal_count}\n")
                lines.append(f"- 成交数: {trade_count}\n")

                # 显示重要事件
                important_events = [e for e in events if e.get('stage') in ['signal_gen', 'order_exec']]
                if important_events:
                    lines.append(f"\n**主要事件**:\n")
                    for event in important_events[:5]:  # 最多显示5个
                        symbol = event.get('symbol', 'SYSTEM')
                        reason = event['reason']
                        lines.append(f"- {symbol}: {reason}\n")

                lines.append("\n")

        if not lines:
            lines.append("暂无历史数据\n")

        return lines


if __name__ == "__main__":
    """自测代码"""
    from events.event_log import EventLogger, Event
    from datetime import datetime

    print("=== DailyPlanner 测试 ===\n")

    # 创建临时日志器并添加测试数据
    logger = EventLogger(log_dir="test_logs")

    # 添加一些历史事件
    yesterday = datetime.now().date() - timedelta(days=1)

    test_events = [
        Event(
            ts=datetime.combine(yesterday, datetime.min.time()),
            symbol="AAPL",
            stage="signal_gen",
            payload={"action": "BUY", "confidence": 0.85},
            reason="金叉买入信号"
        ),
        Event(
            ts=datetime.combine(yesterday, datetime.min.time()),
            symbol="GOOGL",
            stage="signal_gen",
            payload={"action": "HOLD", "confidence": 0.5},
            reason="观望"
        ),
    ]

    for event in test_events:
        logger.log(event)

    # 生成日计划
    planner = DailyPlanner(event_logger=logger)

    target_date = datetime.now().date()
    symbols = ["AAPL", "GOOGL", "MSFT"]

    plan = planner.generate_plan(target_date, symbols)

    print(plan)

    # 清理
    logger.close()

    import shutil
    import os
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")

    print("\n✓ 测试完成")
