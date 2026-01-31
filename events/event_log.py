"""
事件日志系统

使用JSONL格式记录所有系统事件，用于日计划和日复盘生成
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class Event:
    """事件数据结构

    Attributes:
        ts: 时间戳
        symbol: 标的代码（None表示系统级事件）
        stage: 阶段标识（data_fetch, signal_gen, risk_check, order_submit, order_fill, order_reject等）
        payload: 事件数据
        reason: 原因说明
    """
    ts: datetime
    symbol: Optional[str]
    stage: str
    payload: Dict[str, Any]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "ts": self.ts.isoformat(),
            "symbol": self.symbol,
            "stage": self.stage,
            "payload": self.payload,
            "reason": self.reason
        }


class EventLogger:
    """事件日志管理器

    功能：
    - 记录所有系统事件到JSONL文件
    - 按日期分割日志文件
    - 查询事件历史
    """

    def __init__(self, log_dir: str = "logs"):
        """初始化事件日志

        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_date = None
        self.current_file = None

    def _get_log_path(self, date: datetime) -> Path:
        """获取日志文件路径

        Args:
            date: 日期

        Returns:
            日志文件路径
        """
        filename = f"events_{date.strftime('%Y%m%d')}.jsonl"
        return self.log_dir / filename

    def log(self, event: Event):
        """记录事件

        Args:
            event: 事件对象
        """
        # 检查是否需要切换日志文件
        if self.current_date != event.ts.date():
            self.current_date = event.ts.date()
            self.current_file = open(self._get_log_path(event.ts), 'a')

        # 写入事件（JSONL格式）
        if self.current_file is None:
            self.current_file = open(self._get_log_path(event.ts), 'a')

        json.dump(event.to_dict(), self.current_file, ensure_ascii=False)
        self.current_file.write('\n')
        self.current_file.flush()

    def log_event(
        self,
        ts: datetime,
        symbol: Optional[str],
        stage: str,
        payload: Dict[str, Any],
        reason: str
    ):
        """便捷方法：记录事件

        Args:
            ts: 时间戳
            symbol: 标的代码
            stage: 阶段
            payload: 数据
            reason: 原因
        """
        event = Event(
            ts=ts,
            symbol=symbol,
            stage=stage,
            payload=payload,
            reason=reason
        )
        self.log(event)

    def query_events(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        symbol: Optional[str] = None,
        stage: Optional[str] = None
    ) -> list[Dict[str, Any]]:
        """查询事件

        Args:
            start: 开始时间
            end: 结束时间
            symbol: 标的代码
            stage: 阶段

        Returns:
            事件列表
        """
        events = []

        # 确定需要查询的日期范围
        if start and end:
            current = start.date()
            end_date = end.date()
        else:
            # 如果没有指定范围，查询最近7天
            current = datetime.now().date()
            end_date = current
            start = datetime.combine(current, datetime.min.time())
            end = datetime.combine(end_date, datetime.max.time())

        while current <= end_date:
            log_path = self._get_log_path(current)

            if log_path.exists():
                with open(log_path, 'r') as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                            event_ts = datetime.fromisoformat(event['ts'])

                            # 过滤条件
                            if start and event_ts < start:
                                continue
                            if end and event_ts > end:
                                continue
                            if symbol and event.get('symbol') != symbol:
                                continue
                            if stage and event.get('stage') != stage:
                                continue

                            events.append(event)
                        except (json.JSONDecodeError, ValueError) as e:
                            # 跳过无效行
                            continue

            current += timedelta(days=1)

        return events

    def get_summary(self, date: datetime) -> Dict[str, Any]:
        """获取日期摘要

        Args:
            date: 日期

        Returns:
            摘要数据
        """
        events = self.query_events(
            start=datetime.combine(date, datetime.min.time()),
            end=datetime.combine(date, datetime.max.time())
        )

        summary = {
            "date": date.strftime('%Y-%m-%d'),
            "total_events": len(events),
            "by_stage": {},
            "by_symbol": {},
            "first_event": None,
            "last_event": None
        }

        for event in events:
            # 按stage统计
            stage = event.get('stage', 'unknown')
            summary['by_stage'][stage] = summary['by_stage'].get(stage, 0) + 1

            # 按symbol统计
            symbol = event.get('symbol', 'system')
            summary['by_symbol'][symbol] = summary['by_symbol'].get(symbol, 0) + 1

            # 记录首末事件
            if not summary['first_event']:
                summary['first_event'] = event
            summary['last_event'] = event

        return summary

    def close(self):
        """关闭日志文件"""
        if self.current_file:
            self.current_file.close()
            self.current_file = None


# 全局事件日志实例
_global_logger: Optional[EventLogger] = None


def get_event_logger() -> EventLogger:
    """获取全局事件日志实例

    Returns:
        EventLogger实例
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = EventLogger()
    return _global_logger


if __name__ == "__main__":
    """自测代码"""
    from datetime import timedelta

    print("=== EventLogger 测试 ===\n")

    # 创建日志器
    logger = EventLogger(log_dir="test_logs")

    # 测试1: 记录事件
    print("1. 记录测试事件:")
    test_events = [
        Event(
            ts=datetime.now(),
            symbol="AAPL",
            stage="data_fetch",
            payload={"bars_count": 21, "interval": "1d"},
            reason="获取历史K线数据"
        ),
        Event(
            ts=datetime.now(),
            symbol="AAPL",
            stage="signal_gen",
            payload={"action": "BUY", "price": 150.0, "confidence": 0.85},
            reason="金叉买入信号"
        ),
        Event(
            ts=datetime.now(),
            symbol="AAPL",
            stage="risk_check",
            payload={"approved": True, "risk_rules": ["position_size", "fund_sufficiency"]},
            reason="通过风控检查"
        ),
        Event(
            ts=datetime.now(),
            symbol="AAPL",
            stage="order_fill",
            payload={"trade_id": "T001", "side": "BUY", "quantity": 100},
            reason="订单成交"
        ),
        Event(
            ts=datetime.now(),
            symbol=None,
            stage="system",
            payload={"status": "started", "cash": 100000},
            reason="系统启动"
        )
    ]

    for event in test_events:
        logger.log(event)
        print(f"   ✓ {event.ts.strftime('%H:%M:%S')} [{event.stage}] {event.symbol or 'SYSTEM'} - {event.reason}")

    # 测试2: 查询事件
    print("\n2. 查询所有事件:")
    all_events = logger.query_events()
    print(f"   找到 {len(all_events)} 个事件")

    # 测试3: 按symbol过滤
    print("\n3. 查询AAPL的事件:")
    aapl_events = logger.query_events(symbol="AAPL")
    print(f"   找到 {len(aapl_events)} 个事件")
    for event in aapl_events:
        print(f"   {event['ts']}: [{event['stage']}] {event['reason']}")

    # 测试4: 按stage过滤
    print("\n4. 查询signal_gen阶段的事件:")
    signal_events = logger.query_events(stage="signal_gen")
    print(f"   找到 {len(signal_events)} 个事件")
    for event in signal_events:
        print(f"   {event['symbol']}: {event['payload']['action']} @ {event['payload']['price']}")

    # 测试5: 获取摘要
    print("\n5. 日期摘要:")
    summary = logger.get_summary(datetime.now().date())
    print(f"   总事件数: {summary['total_events']}")
    print(f"   按阶段统计:")
    for stage, count in summary['by_stage'].items():
        print(f"     {stage}: {count}")
    print(f"   按标的统计:")
    for symbol, count in summary['by_symbol'].items():
        print(f"     {symbol}: {count}")

    # 测试6: 读取JSONL文件
    print("\n6. 验证JSONL文件:")
    log_path = logger._get_log_path(datetime.now().date())
    print(f"   文件路径: {log_path}")
    print(f"   文件存在: {log_path.exists()}")

    if log_path.exists():
        with open(log_path, 'r') as f:
            lines = f.readlines()
            print(f"   文件行数: {len(lines)}")
            print(f"   前3行:")
            for line in lines[:3]:
                event = json.loads(line)
                print(f"     {event['ts']}: {event['stage']}")

    # 清理
    logger.close()

    # 清理测试文件
    import shutil
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")
        print("\n   ✓ 测试文件已清理")

    print("\n✓ 所有测试通过")
