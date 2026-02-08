"""
审计模块

提供审计记录功能，确保任何关键动作都可追责。
"""

import json
import csv
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class AuditActionType(Enum):
    """审计操作类型"""
    # 策略相关
    STRATEGY_ENABLE = "strategy_enable"
    STRATEGY_DISABLE = "strategy_disable"
    STRATEGY_CONFIG_UPDATE = "strategy_config_update"

    # 风控相关
    RISK_TAKEOVER = "risk_takeover"
    RISK_RULE_UPDATE = "risk_rule_update"
    RISK_LIMIT_UPDATE = "risk_limit_update"

    # 配置相关
    LIVE_LOCK = "live_lock"
    LIVE_UNLOCK = "live_unlock"
    RULE_UPDATE = "rule_update"
    CONFIG_UPDATE = "config_update"

    # 其他
    MANUAL_ORDER = "manual_order"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class AuditRecord:
    """审计记录"""
    actor: str                      # 操作者
    action: AuditActionType          # 操作类型
    reason: str                      # 操作原因
    target_id: Optional[str] = None  # 目标 ID
    target_type: Optional[str] = None  # 目标类型 (strategy/portfolio/config)
    approver: Optional[str] = None   # 批准人（某些操作必需）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
    timestamp: datetime = field(default_factory=datetime.now)
    trace_id: Optional[str] = None   # 追踪 ID

    def __post_init__(self):
        """生成 trace_id"""
        if self.trace_id is None:
            # 生成唯一的 trace_id
            timestamp_str = self.timestamp.isoformat()
            content = f"{self.actor}:{self.action.value}:{timestamp_str}"
            hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
            self.trace_id = f"trace_{hash_val}"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "actor": self.actor,
            "action": self.action.value if isinstance(self.action, AuditActionType) else self.action,
            "reason": self.reason,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "approver": self.approver,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id
        }


class AuditLog:
    """审计日志"""

    # 需要 approver 的操作
    REQUIRES_APPROVER = {
        AuditActionType.LIVE_UNLOCK,
    }

    def __init__(self, storage_path: str = "audit.jsonl"):
        """初始化审计日志

        Args:
            storage_path: 存储文件路径（JSONL 格式）
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        actor: str,
        action: AuditActionType,
        reason: str,
        target_id: Optional[str] = None,
        target_type: Optional[str] = None,
        approver: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditRecord:
        """记录审计事件

        Args:
            actor: 操作者
            action: 操作类型
            reason: 操作原因
            target_id: 目标 ID
            target_type: 目标类型
            approver: 批准人
            metadata: 额外元数据

        Returns:
            AuditRecord
        """
        # 检查需要 approver 的操作
        if action in self.REQUIRES_APPROVER and approver is None:
            raise ValueError(f"Action {action.value} requires an approver")

        record = AuditRecord(
            actor=actor,
            action=action,
            reason=reason,
            target_id=target_id,
            target_type=target_type,
            approver=approver,
            metadata=metadata or {}
        )

        # 写入文件
        self._append_record(record)

        return record

    def _append_record(self, record: AuditRecord) -> None:
        """追加记录到文件

        Args:
            record: 审计记录
        """
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(record.to_dict()) + '\n')

    def get_records(
        self,
        limit: int = 100,
        actor: Optional[str] = None,
        action: Optional[AuditActionType] = None,
        target_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AuditRecord]:
        """获取审计记录

        Args:
            limit: 返回数量
            actor: 过滤操作者
            action: 过滤操作类型
            target_id: 过滤目标 ID
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            审计记录列表（按时间倒序）
        """
        records = []

        if not self.storage_path.exists():
            return records

        with open(self.storage_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    record = self._dict_to_record(data)

                    # 应用过滤
                    if actor and record.actor != actor:
                        continue
                    if action and record.action != action:
                        continue
                    if target_id and record.target_id != target_id:
                        continue
                    if start_time and record.timestamp < start_time:
                        continue
                    if end_time and record.timestamp > end_time:
                        continue

                    records.append(record)

                except (json.JSONDecodeError, ValueError):
                    continue

        # 按时间倒序
        records.sort(key=lambda r: r.timestamp, reverse=True)

        return records[:limit]

    def _dict_to_record(self, data: Dict[str, Any]) -> AuditRecord:
        """将字典转换为 AuditRecord

        Args:
            data: 字典数据

        Returns:
            AuditRecord
        """
        # 转换 action 字符串为枚举
        action_str = data.get("action")
        try:
            action = AuditActionType(action_str)
        except ValueError:
            action = action_str

        # 转换 timestamp
        timestamp_str = data.get("timestamp")
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str)
        else:
            timestamp = datetime.now()

        return AuditRecord(
            actor=data["actor"],
            action=action,
            reason=data["reason"],
            target_id=data.get("target_id"),
            target_type=data.get("target_type"),
            approver=data.get("approver"),
            metadata=data.get("metadata", {}),
            timestamp=timestamp,
            trace_id=data.get("trace_id")
        )

    def export_json(self, output_path: str) -> int:
        """导出为 JSON

        Args:
            output_path: 输出文件路径

        Returns:
            导出的记录数
        """
        records = self.get_records(limit=10000)  # 获取所有记录

        with open(output_path, 'w') as f:
            json.dump([r.to_dict() for r in records], f, indent=2)

        return len(records)

    def export_csv(self, output_path: str) -> int:
        """导出为 CSV

        Args:
            output_path: 输出文件路径

        Returns:
            导出的记录数
        """
        records = self.get_records(limit=10000)  # 获取所有记录

        if not records:
            return 0

        # 获取所有字段
        fieldnames = ["actor", "action", "reason", "target_id", "target_type", "approver", "timestamp", "trace_id"]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for record in records:
                row = {
                    "actor": record.actor,
                    "action": record.action.value if isinstance(record.action, AuditActionType) else record.action,
                    "reason": record.reason,
                    "target_id": record.target_id,
                    "target_type": record.target_type,
                    "approver": record.approver,
                    "timestamp": record.timestamp.isoformat(),
                    "trace_id": record.trace_id
                }
                writer.writerow(row)

        return len(records)


# 全局审计日志实例
_global_audit_log: Optional[AuditLog] = None


def get_audit_log(storage_path: str = "audit.jsonl") -> AuditLog:
    """获取全局审计日志实例

    Args:
        storage_path: 存储路径

    Returns:
        AuditLog 实例
    """
    global _global_audit_log

    if _global_audit_log is None:
        _global_audit_log = AuditLog(storage_path)

    return _global_audit_log


def log_strategy_disable(
    actor: str,
    strategy_id: str,
    reason: str
) -> AuditRecord:
    """记录策略禁用

    Args:
        actor: 操作者
        strategy_id: 策略 ID
        reason: 原因

    Returns:
        AuditRecord
    """
    audit_log = get_audit_log()
    return audit_log.log(
        actor=actor,
        action=AuditActionType.STRATEGY_DISABLE,
        reason=reason,
        target_id=strategy_id,
        target_type="strategy"
    )


def log_risk_takeover(
    actor: str,
    portfolio_id: str,
    reason: str,
    metadata: Optional[Dict[str, Any]] = None
) -> AuditRecord:
    """记录风控接管

    Args:
        actor: 操作者（通常是 "system"）
        portfolio_id: 组合 ID
        reason: 原因
        metadata: 额外信息（如 drawdown, limit 等）

    Returns:
        AuditRecord
    """
    audit_log = get_audit_log()
    return audit_log.log(
        actor=actor,
        action=AuditActionType.RISK_TAKEOVER,
        reason=reason,
        target_id=portfolio_id,
        target_type="portfolio",
        metadata=metadata or {}
    )


def log_live_unlock(
    actor: str,
    reason: str,
    approver: str
) -> AuditRecord:
    """记录 live 模式解锁

    Args:
        actor: 操作者
        reason: 原因
        approver: 批准人（必需）

    Returns:
        AuditRecord
    """
    audit_log = get_audit_log()
    return audit_log.log(
        actor=actor,
        action=AuditActionType.LIVE_UNLOCK,
        reason=reason,
        target_id="live_mode",
        target_type="config",
        approver=approver
    )
