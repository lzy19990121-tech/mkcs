"""
SPL-7a-E: 在线→离线桥接（Risk Event Store）

将所有在线风险事件写入持久化存储，供 SPL-7b 和 SPL-6a 使用。
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import sqlite3
from collections import deque

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.online.risk_signal_schema import RiskSignal, PortfolioRiskSignal, GatingEvent, AllocatorEvent
from analysis.online.risk_state_machine import RiskState, StateTransitionEvent
from analysis.online.trend_detector import TrendAlert
from analysis.online.postmortem_generator import PostMortemReport


class EventType(Enum):
    """事件类型"""
    RISK_SIGNAL = "risk_signal"                    # 风险信号
    STATE_TRANSITION = "state_transition"          # 状态转换
    TREND_ALERT = "trend_alert"                    # 趋势告警
    GATING_EVENT = "gating_event"                  # Gating 事件
    ALLOCATOR_EVENT = "allocator_event"            # Allocator 事件
    POST_MORTEM = "post_mortem"                    # Post-mortem 报告


@dataclass
class RiskEvent:
    """风险事件（统一格式）"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    strategy_id: str

    # 事件数据
    data: Dict[str, Any]

    # 元信息
    data_version: str = "1.0"
    source: str = "online_monitoring"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "strategy_id": self.strategy_id,
            "data": self.data,
            "data_version": self.data_version,
            "source": self.source
        }

    @classmethod
    def from_signal(cls, signal: RiskSignal) -> "RiskEvent":
        """从风险信号创建事件"""
        return cls(
            event_id=f"signal_{signal.strategy_id}_{signal.timestamp.strftime('%Y%m%d%H%M%S')}",
            event_type=EventType.RISK_SIGNAL,
            timestamp=signal.timestamp,
            strategy_id=signal.strategy_id,
            data=signal.to_dict()
        )

    @classmethod
    def from_state_transition(cls, transition: StateTransitionEvent) -> "RiskEvent":
        """从状态转换创建事件"""
        return cls(
            event_id=transition.to_dict().get("event_id", ""),
            event_type=EventType.STATE_TRANSITION,
            timestamp=transition.timestamp,
            strategy_id=transition.strategy_id,
            data=transition.to_dict()
        )

    @classmethod
    def from_trend_alert(cls, alert: TrendAlert) -> "RiskEvent":
        """从趋势告警创建事件"""
        return cls(
            event_id=alert.to_dict().get("alert_id", ""),
            event_type=EventType.TREND_ALERT,
            timestamp=alert.timestamp,
            strategy_id=alert.strategy_id,
            data=alert.to_dict()
        )

    @classmethod
    def from_post_mortem(cls, report: PostMortemReport) -> "RiskEvent":
        """从 post-mortem 创建事件"""
        return cls(
            event_id=report.report_id,
            event_type=EventType.POST_MORTEM,
            timestamp=report.trigger_time,
            strategy_id=report.strategy_id,
            data=report.to_dict()
        )


class RiskEventStore:
    """风险事件存储

    持久化所有风险事件，供离线分析使用。
    """

    def __init__(
        self,
        db_path: str = "data/risk_events.db",
        enable_wal: bool = True
    ):
        """初始化事件存储

        Args:
            db_path: 数据库路径
            enable_wal: 是否启用 WAL（Write-Ahead Logging）
        """
        self.db_path = db_path
        self.enable_wal = enable_wal

        # 创建数据库
        self._init_database()

        # 内存缓冲（批量写入）
        self.write_buffer: deque = deque(maxlen=1000)
        self.buffer_size = 100

    def _init_database(self):
        """初始化数据库表"""
        # 确保目录存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 启用 WAL
        if self.enable_wal:
            cursor.execute("PRAGMA journal_mode=WAL")

        # 创建事件表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS risk_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                data TEXT NOT NULL,
                data_version TEXT,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type_timestamp
            ON risk_events(event_type, timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_strategy_timestamp
            ON risk_events(strategy_id, timestamp)
        """)

        conn.commit()
        conn.close()

    def write_event(self, event: RiskEvent) -> bool:
        """写入单个事件

        Args:
            event: 风险事件

        Returns:
            是否成功
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO risk_events
                (event_id, event_type, timestamp, strategy_id, data, data_version, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type.value,
                event.timestamp.isoformat(),
                event.strategy_id,
                json.dumps(event.data),
                event.data_version,
                event.source
            ))

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"写入事件失败: {e}")
            return False

    def write_events_batch(self, events: List[RiskEvent]) -> int:
        """批量写入事件

        Args:
            events: 事件列表

        Returns:
            成功写入的数量
        """
        if not events:
            return 0

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            success_count = 0

            for event in events:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO risk_events
                        (event_id, event_type, timestamp, strategy_id, data, data_version, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type.value,
                        event.timestamp.isoformat(),
                        event.strategy_id,
                        json.dumps(event.data),
                        event.data_version,
                        event.source
                    ))
                    success_count += 1

                except Exception as e:
                    print(f"写入事件 {event.event_id} 失败: {e}")

            conn.commit()
            conn.close()

            return success_count

        except Exception as e:
            print(f"批量写入失败: {e}")
            return 0

    def query_events(
        self,
        event_type: Optional[EventType] = None,
        strategy_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """查询事件

        Args:
            event_type: 事件类型过滤
            strategy_id: 策略 ID 过滤
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量限制

        Returns:
            事件列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 构建查询
            query = "SELECT * FROM risk_events WHERE 1=1"
            params = []

            if event_type:
                query += " AND event_type = ?"
                params.append(event_type.value)

            if strategy_id:
                query += " AND strategy_id = ?"
                params.append(strategy_id)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # 转换为字典列表
            columns = [desc[0] for desc in cursor.description]
            events = [dict(zip(columns, row)) for row in rows]

            conn.close()

            return events

        except Exception as e:
            print(f"查询事件失败: {e}")
            return []

    def get_event_count(
        self,
        event_type: Optional[EventType] = None,
        strategy_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """获取事件数量

        Args:
            event_type: 事件类型过滤
            strategy_id: 策略 ID 过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            事件数量
        """
        events = self.query_events(
            event_type=event_type,
            strategy_id=strategy_id,
            start_time=start_time,
            end_time=end_time,
            limit=1000000  # 大数量
        )

        return len(events)

    def get_latest_events(
        self,
        strategy_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取最新事件

        Args:
            strategy_id: 策略 ID
            limit: 返回数量

        Returns:
            事件列表
        """
        return self.query_events(
            strategy_id=strategy_id,
            limit=limit
        )

    def get_events_for_counterfactual(
        self,
        strategy_id: str,
        event_id: str,
        context_window_hours: int = 24
    ) -> Dict[str, Any]:
        """获取用于反事实分析的事件上下文

        Args:
            strategy_id: 策略 ID
            event_id: 事件 ID
            context_window_hours: 上下文窗口（小时）

        Returns:
            事件上下文
        """
        # 查找事件
        events = self.query_events(
            strategy_id=strategy_id,
            limit=1000
        )

        # 找到目标事件
        target_event = None
        for event in events:
            if event["event_id"] == event_id:
                target_event = event
                break

        if not target_event:
            return {}

        # 提取上下文窗口
        event_time = datetime.fromisoformat(target_event["timestamp"])
        start_time = event_time - timedelta(hours=context_window_hours)
        end_time = event_time + timedelta(hours=context_window_hours)

        context_events = [
            e for e in events
            if start_time <= datetime.fromisoformat(e["timestamp"]) <= end_time
        ]

        return {
            "target_event": target_event,
            "context_events": context_events,
            "window_start": start_time.isoformat(),
            "window_end": end_time.isoformat()
        }

    def export_to_json(
        self,
        output_path: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> bool:
        """导出事件到 JSON

        Args:
            output_path: 输出文件路径
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            是否成功
        """
        try:
            events = self.query_events(
                start_time=start_time,
                end_time=end_time,
                limit=1000000
            )

            with open(output_path, 'w') as f:
                json.dump(events, f, indent=2, default=str)

            print(f"导出 {len(events)} 个事件到 {output_path}")
            return True

        except Exception as e:
            print(f"导出失败: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计

        Returns:
            统计信息
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 总事件数
            cursor.execute("SELECT COUNT(*) FROM risk_events")
            total_count = cursor.fetchone()[0]

            # 按类型分组
            cursor.execute("""
                SELECT event_type, COUNT(*) as count
                FROM risk_events
                GROUP BY event_type
            """)
            by_type = dict(cursor.fetchall())

            # 按策略分组
            cursor.execute("""
                SELECT strategy_id, COUNT(*) as count
                FROM risk_events
                GROUP BY strategy_id
            """)
            by_strategy = dict(cursor.fetchall())

            # 时间范围
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM risk_events")
            min_time, max_time = cursor.fetchone()

            conn.close()

            return {
                "total_events": total_count,
                "by_type": by_type,
                "by_strategy": by_strategy,
                "time_range": {
                    "earliest": min_time,
                    "latest": max_time
                }
            }

        except Exception as e:
            print(f"获取统计失败: {e}")
            return {}


class RiskEventBridge:
    """在线→离线桥接器

    自动将在线事件写入存储。
    """

    def __init__(
        self,
        strategy_id: str,
        event_store: Optional[RiskEventStore] = None
    ):
        """初始化桥接器

        Args:
            strategy_id: 策略 ID
            event_store: 事件存储（可选，默认创建新实例）
        """
        self.strategy_id = strategy_id
        self.event_store = event_store or RiskEventStore()

    def bridge_signal(self, signal: RiskSignal) -> bool:
        """桥接风险信号

        Args:
            signal: 风险信号

        Returns:
            是否成功
        """
        event = RiskEvent.from_signal(signal)
        return self.event_store.write_event(event)

    def bridge_state_transition(self, transition: StateTransitionEvent) -> bool:
        """桥接状态转换

        Args:
            transition: 状态转换

        Returns:
            是否成功
        """
        event = RiskEvent.from_state_transition(transition)
        return self.event_store.write_event(event)

    def bridge_trend_alert(self, alert: TrendAlert) -> bool:
        """桥接趋势告警

        Args:
            alert: 趋势告警

        Returns:
            是否成功
        """
        event = RiskEvent.from_trend_alert(alert)
        return self.event_store.write_event(event)

    def bridge_post_mortem(self, report: PostMortemReport) -> bool:
        """桥接 post-mortem

        Args:
            report: Post-mortem 报告

        Returns:
            是否成功
        """
        event = RiskEvent.from_post_mortem(report)
        return self.event_store.write_event(event)
