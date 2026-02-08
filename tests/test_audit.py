"""
P1-4: 审计信息完善

实现审计记录功能，确保任何关键动作都可追责：
- 定义 AuditRecord schema：actor、approver、action、reason、timestamp、hash/trace_id
- 在关键路径写入：rules 更新、策略启停、风控接管、live 模式开关
- UI/报告中展示审计字段

验收:
☐ 任意一次"策略禁用/风险接管"都有 actor+reason
☐ 任意一次"live 解锁"必须有 approver 字段
☐ 审计记录可导出（JSON/CSV）
"""

import sys
import os
from pathlib import Path
import tempfile
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from analysis.audit import AuditRecord, AuditLog, AuditActionType


def test_audit_record_schema():
    """测试1: AuditRecord schema"""
    print("\n" + "="*70)
    print("测试1: AuditRecord schema")
    print("="*70)

    record = AuditRecord(
        actor="user_123",
        action=AuditActionType.STRATEGY_DISABLE,
        reason="策略表现不佳，暂时禁用",
        target_id="strategy_ma",
        target_type="strategy",
        approver="admin_456",
        metadata={"performance": -0.05}
    )

    # 检查所有必需字段
    assert record.actor == "user_123", "actor 应该存在"
    assert record.action == AuditActionType.STRATEGY_DISABLE, "action 应该存在"
    assert record.reason is not None, "reason 应该存在"
    assert record.trace_id is not None, "trace_id 应该自动生成"
    assert record.timestamp is not None, "timestamp 应该自动生成"

    # 检查 to_dict
    record_dict = record.to_dict()
    assert "actor" in record_dict, "to_dict 应该包含 actor"
    assert "action" in record_dict, "to_dict 应该包含 action"
    assert "approver" in record_dict, "to_dict 应该包含 approver"

    print(f"  ✅ AuditRecord schema 完整")
    print(f"  - actor: {record.actor}")
    print(f"  - action: {record.action.value}")
    print(f"  - reason: {record.reason}")
    print(f"  - approver: {record.approver}")
    print(f"  - trace_id: {record.trace_id}")

    return True


def test_audit_log_operations():
    """测试2: 审计日志操作"""
    print("\n" + "="*70)
    print("测试2: 审计日志操作")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audit_log = AuditLog(storage_path=str(temp_path / "audit.jsonl"))

        # 记录审计事件
        audit_log.log(
            actor="user_123",
            action=AuditActionType.STRATEGY_DISABLE,
            reason="策略表现不佳",
            target_id="strategy_ma",
            target_type="strategy"
        )

        # 风控接管
        audit_log.log(
            actor="system",
            action=AuditActionType.RISK_TAKEOVER,
            reason="回撤超过限制",
            target_id="portfolio_1",
            target_type="portfolio",
            metadata={"drawdown": 0.15, "limit": 0.10}
        )

        # live 解锁（需要 approver）
        audit_log.log(
            actor="trader_001",
            action=AuditActionType.LIVE_UNLOCK,
            reason="需要紧急调整",
            target_id="live_mode",
            target_type="config",
            approver="admin_456"
        )

        # 获取所有记录（按时间倒序）
        records = audit_log.get_records(limit=10)

        assert len(records) == 3, "应该有 3 条记录"
        # 由于是倒序，最后添加的在最前面
        assert records[0].action == AuditActionType.LIVE_UNLOCK, "第一条（倒序）应该是 live 解锁"
        assert records[1].action == AuditActionType.RISK_TAKEOVER, "第二条应该是风控接管"
        assert records[2].action == AuditActionType.STRATEGY_DISABLE, "第三条应该是策略禁用"
        assert records[0].approver == "admin_456", "live 解锁应该有 approver"

        print(f"  ✅ 审计日志操作正确")
        print(f"  - 总记录数: {len(records)}")
        for i, r in enumerate(records):
            print(f"  - {i+1}. {r.action.value} by {r.actor}")

    return True


def test_audit_filtering():
    """测试3: 审计记录过滤"""
    print("\n" + "="*70)
    print("测试3: 审计记录过滤")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audit_log = AuditLog(storage_path=str(temp_path / "audit.jsonl"))

        # 添加不同类型的记录
        audit_log.log(
            actor="user_1",
            action=AuditActionType.STRATEGY_DISABLE,
            reason="test",
            target_id="s1"
        )
        audit_log.log(
            actor="user_2",
            action=AuditActionType.RISK_TAKEOVER,
            reason="test",
            target_id="p1"
        )
        audit_log.log(
            actor="user_1",
            action=AuditActionType.LIVE_UNLOCK,
            reason="test",
            target_id="live",
            approver="admin"
        )

        # 按 actor 过滤
        records = audit_log.get_records(actor="user_1")
        assert len(records) == 2, "user_1 应该有 2 条记录"

        # 按 action 过滤
        records = audit_log.get_records(action=AuditActionType.RISK_TAKEOVER)
        assert len(records) == 1, "应该有 1 条风控接管记录"

        # 按时间范围过滤
        from datetime import timedelta
        start = datetime.now() - timedelta(hours=1)
        records = audit_log.get_records(start_time=start)
        assert len(records) == 3, "最近一小时内应该有 3 条记录"

        print(f"  ✅ 审计记录过滤正确")
        print(f"  - 按 actor 过滤: user_1 有 2 条")
        print(f"  - 按 action 过滤: RISK_TAKEOVER 有 1 条")
        print(f"  - 按时间过滤: 最近一小时有 3 条")

    return True


def test_export_json():
    """测试4: 导出 JSON"""
    print("\n" + "="*70)
    print("测试4: 导出 JSON")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audit_log = AuditLog(storage_path=str(temp_path / "audit.jsonl"))

        # 添加记录
        audit_log.log(
            actor="user_1",
            action=AuditActionType.STRATEGY_ENABLE,
            reason="重新启用策略",
            target_id="strategy_ma"
        )

        # 导出 JSON
        output_file = temp_path / "export.json"
        count = audit_log.export_json(str(output_file))

        assert count > 0, "应该导出至少 1 条记录"
        assert output_file.exists(), "导出文件应该存在"

        # 验证导出内容
        with open(output_file) as f:
            data = json.load(f)
            assert len(data) == count, "导出的记录数应该匹配"
            assert "actor" in data[0], "应该包含 actor 字段"
            assert "action" in data[0], "应该包含 action 字段"

        print(f"  ✅ 导出 JSON 正确")
        print(f"  - 导出记录数: {count}")
        print(f"  - 文件路径: {output_file}")

    return True


def test_export_csv():
    """测试5: 导出 CSV"""
    print("\n" + "="*70)
    print("测试5: 导出 CSV")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audit_log = AuditLog(storage_path=str(temp_path / "audit.jsonl"))

        # 添加记录
        audit_log.log(
            actor="user_1",
            action=AuditActionType.RULE_UPDATE,
            reason="更新风险参数",
            target_id="rule_1"
        )
        audit_log.log(
            actor="user_2",
            action=AuditActionType.STRATEGY_DISABLE,
            reason="测试",
            target_id="s2"
        )

        # 导出 CSV
        output_file = temp_path / "export.csv"
        count = audit_log.export_csv(str(output_file))

        assert count > 0, "应该导出至少 1 条记录"
        assert output_file.exists(), "导出文件应该存在"

        # 验证导出内容
        with open(output_file) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == count, "导出的记录数应该匹配"
            assert "actor" in rows[0], "应该包含 actor 列"
            assert "action" in rows[0], "应该包含 action 列"

        print(f"  ✅ 导出 CSV 正确")
        print(f"  - 导出记录数: {count}")
        print(f"  - 文件路径: {output_file}")

    return True


def test_required_approver():
    """测试6: 必需 approver 的操作"""
    print("\n" + "="*70)
    print("测试6: 必需 approver 的操作")
    print("="*70)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        audit_log = AuditLog(storage_path=str(temp_path / "audit.jsonl"))

        # LIVE_UNLOCK 需要 approver
        try:
            audit_log.log(
                actor="user_1",
                action=AuditActionType.LIVE_UNLOCK,
                reason="测试",
                target_id="live"
                # 故意不提供 approver
            )
            assert False, "应该抛出异常"
        except ValueError as e:
            assert "approver" in str(e).lower(), "应该提示需要 approver"

        print(f"  ✅ LIVE_UNLOCK 正确要求 approver")

        # 提供 approver 后应该成功
        audit_log.log(
            actor="user_1",
            action=AuditActionType.LIVE_UNLOCK,
            reason="测试",
            target_id="live",
            approver="admin"
        )

        records = audit_log.get_records()
        assert len(records) == 1, "应该有 1 条记录"
        assert records[0].approver == "admin", "应该记录 approver"

        print(f"  ✅ 提供 approver 后成功记录")

    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("P1-4: 审计信息完善测试")
    print("="*70)

    tests = [
        ("AuditRecord schema", test_audit_record_schema),
        ("审计日志操作", test_audit_log_operations),
        ("审计记录过滤", test_audit_filtering),
        ("导出 JSON", test_export_json),
        ("导出 CSV", test_export_csv),
        ("必需 approver 的操作", test_required_approver),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"  ❌ {name} 失败: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("="*70)

    if failed == 0:
        print("\n🎉 所有测试通过！")

    sys.exit(0 if failed == 0 else 1)
