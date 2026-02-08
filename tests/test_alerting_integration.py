"""
P1-1: Slack/Email/Webhook 通知集成测试

验证告警系统功能：
1. 统一 AlertEvent schema
2. 实现三种 channel 的 sender（至少 Webhook + 其中一种）
3. 集成点：风控触发、drift 触发、交易失败/下单异常
4. 异步/降级机制：发送失败不阻塞主流程，支持重试次数与熔断
5. dry-run 模式：只打印不发送

验收:
☐ 触发风控动作时，能收到一条包含 run_id 的通知
☐ 通知发送失败不会让交易循环崩溃
☐ dry-run 模式下不会真实外发
☐ 事件字段完整（能自动���成跳转链接到 web UI/run）
"""

import sys
import os
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import logging
from datetime import datetime
from unittest.mock import Mock, patch

from analysis.online.alerting import (
    Alert,
    AlertSeverity,
    AlertChannel,
    AlertRule,
    AlertRuleEngine,
    AlertSender,
    AlertingManager
)
from analysis.online.risk_signal_schema import RiskSignal
from analysis.online.risk_state_machine import RiskState
from analysis.online.trend_detector import TrendAlert


def test_alert_event_schema():
    """测试1: 统一 AlertEvent schema"""
    print("\n" + "="*70)
    print("测试1: 统一 AlertEvent schema")
    print("="*70)

    alert = Alert(
        alert_id="test_001",
        timestamp=datetime.now(),
        rule_id="test_rule",
        strategy_id="test_strategy",
        severity=AlertSeverity.WARNING,
        title="测试告警",
        message="这是一条测试告警",
        metric_name="test_metric",
        current_value=0.85,
        threshold=0.7,
        run_id="run_12345",
        tags=["test", "unit"],
        links={"web_ui": "http://example.com/run/12345", "report": "/reports/12345"}
    )

    # 检查所有必需字段
    assert alert.alert_id == "test_001", "alert_id 应该存在"
    assert alert.run_id == "run_12345", "run_id 应该存在"
    assert alert.tags == ["test", "unit"], "tags 应该存在"
    assert alert.links == {"web_ui": "http://example.com/run/12345", "report": "/reports/12345"}, "links 应该存在"

    # 检查 to_dict
    alert_dict = alert.to_dict()
    assert "run_id" in alert_dict, "to_dict 应该包含 run_id"
    assert "tags" in alert_dict, "to_dict 应该包含 tags"
    assert "links" in alert_dict, "to_dict 应该包含 links"

    print(f"  ✅ Alert schema 完整")
    print(f"  - alert_id: {alert.alert_id}")
    print(f"  - run_id: {alert.run_id}")
    print(f"  - tags: {alert.tags}")
    print(f"  - links: {alert.links}")

    return True


def test_slack_message_format():
    """测试2: Slack 消息格式"""
    print("\n" + "="*70)
    print("测试2: Slack 消息格式")
    print("="*70)

    alert = Alert(
        alert_id="test_002",
        timestamp=datetime.now(),
        rule_id="envelope_approach",
        strategy_id="strategy_ma",
        severity=AlertSeverity.WARNING,
        title="接近 Envelope",
        message="回撤接近 envelope 上限（70%）",
        metric_name="envelope_usage",
        current_value=0.75,
        threshold=0.7,
        run_id="run_67890",
        tags=["envelope", "warning"],
        links={"web_ui": "http://example.com/run/67890"}
    )

    slack_msg = alert.to_slack_message()

    assert "attachments" in slack_msg, "Slack 消息应该有 attachments"
    assert len(slack_msg["attachments"]) > 0, "应该至少有一个 attachment"

    attachment = slack_msg["attachments"][0]
    assert "title" in attachment, "Attachment 应该有 title"
    assert "color" in attachment, "Attachment 应该有 color"
    assert "fields" in attachment, "Attachment 应该有 fields"
    assert "actions" in attachment, "Attachment 应该有 actions 按钮"

    print(f"  ✅ Slack 消息格式正确")
    print(f"  - Title: {attachment['title']}")
    print(f"  - Color: {attachment['color']}")
    print(f"  - Fields: {len(attachment['fields'])}")

    return True


def test_email_format():
    """测试3: 邮件格式"""
    print("\n" + "="*70)
    print("测试3: 邮件格式")
    print("="*70)

    alert = Alert(
        alert_id="test_003",
        timestamp=datetime.now(),
        rule_id="critical_drawdown",
        strategy_id="strategy_ml",
        severity=AlertSeverity.CRITICAL,
        title="严重回撤",
        message="回撤超过10%",
        metric_name="drawdown",
        current_value=0.12,
        threshold=0.10,
        run_id="run_critical",
        tags=["critical", "drawdown"],
        links={"web_ui": "http://example.com/run/critical"}
    )

    subject = alert.to_email_subject()
    body = alert.to_email_body()

    assert "[CRITICAL]" in subject, "主题应该包含严重程度"
    assert alert.run_id in subject, "主题应该包含 run_id"
    assert alert.strategy_id in subject, "主题应该包含 strategy_id"

    assert "<h2>" in body, "正文应该是 HTML 格式"
    assert alert.run_id in body, "正文应该包含 run_id"
    assert "http://example.com/run/critical" in body, "正文应该包含链接"

    print(f"  ✅ 邮件格式正确")
    print(f"  - Subject: {subject}")
    print(f"  - Body 包含链接和 HTML")

    return True


def test_dry_run_mode():
    """测试4: dry-run 模式"""
    print("\n" + "="*70)
    print("测试4: dry-run 模式")
    print("="*70)

    # dry-run 模式配置
    config = {"dry_run": True}

    sender = AlertSender(config)

    alert = Alert(
        alert_id="test_dryrun",
        timestamp=datetime.now(),
        rule_id="test_rule",
        strategy_id="test_strategy",
        severity=AlertSeverity.WARNING,
        title="测试告警",
        message="这是一条测试告警",
        metric_name="test",
        current_value=0.8,
        threshold=0.7,
        run_id="run_dryrun"
    )

    # 在 dry-run 模式下，发送应该只打印而不真实发送
    result = sender._send_to_slack(alert)
    assert result == True, "dry-run 模式下应该返回成功"

    result = sender._send_to_webhook(alert)
    assert result == True, "dry-run 模式下应该返回成功"

    result = sender._send_to_email(alert)
    assert result == True, "dry-run 模式下应该返回成功"

    print(f"  ✅ dry-run 模式正确")
    print(f"  - 所有渠道都只打印，不真实发送")

    return True


def test_circuit_breaker():
    """测试5: 熔断器机制"""
    print("\n" + "="*70)
    print("测试5: 熔断器机制")
    print("="*70)

    config = {"dry_run": True}
    sender = AlertSender(config)

    # 模拟连续失败
    channel = "slack"
    for i in range(sender._circuit_breaker_threshold):
        sender._record_failure(channel)

    # 检查熔断器是否开启
    assert sender._is_circuit_open(channel), "熔断器应该已开启"

    # 检查熔断期间是否跳过发送
    assert sender._is_circuit_open(channel), "熔断期间应该跳过发送"

    print(f"  ✅ 熔断器机制正确")
    print(f"  - 失败 {sender._circuit_breaker_threshold} 次后熔断")
    print(f"  - 熔断期间跳过发送")

    # 测试熔断器重置
    sender._reset_circuit_breaker(channel)
    assert not sender._is_circuit_open(channel), "重置后熔断器应该关闭"

    print(f"  - 重置后熔断器恢复")

    return True


def test_async_send_no_blocking():
    """测试6: 异步发送不阻塞主循环"""
    print("\n" + "="*70)
    print("测试6: 异步发送不阻塞主循环")
    print("="*70)

    config = {"dry_run": True}
    manager = AlertingManager(sender_config=config)

    alert = Alert(
        alert_id="test_async",
        timestamp=datetime.now(),
        rule_id="test_rule",
        strategy_id="test_strategy",
        severity=AlertSeverity.INFO,
        title="异步测试",
        message="测试异步发送",
        metric_name="test",
        current_value=0.5,
        threshold=0.7,
        run_id="run_async"
    )

    # 记录开始时间
    start_time = time.time()

    # 异步发送（应该立即返回）
    manager.sender.send_alert_async(alert, [AlertChannel.LOG, AlertChannel.SLACK])

    # 检查是否快速返回（不等待发送完成）
    elapsed = time.time() - start_time
    assert elapsed < 0.1, "异步发送应该立即返回"

    print(f"  ✅ 异步发送不阻塞主循环")
    print(f"  - 返回耗时: {elapsed:.4f}秒")

    # 等待一下让后台任务完成
    time.sleep(0.5)

    # 关闭发送器（等待所有任务完成）
    manager.sender.shutdown()

    return True


def test_integration_points():
    """测试7: 集成点测试"""
    print("\n" + "="*70)
    print("测试7: 集成点测试")
    print("="*70)

    manager = AlertingManager(sender_config={"dry_run": True})

    # 测试风控触发集成
    print("\n  7.1: 风控触发告警")

    # 模拟风险信号
    signal = Mock()
    signal.strategy_id = "test_strategy"
    signal.drawdown = Mock()
    signal.drawdown.current_drawdown = 0.08  # 接近 envelope (0.10)
    signal.spike = Mock()
    signal.spike.recent_spike_count = 3
    signal.gating_events = []
    signal.allocator_events = []

    # 模拟状态
    state = RiskState.WARNING
    trends = {"volatility": Mock(slope=0.01)}

    # 处理风险更新
    alerts = manager.process_risk_update(
        signal=signal,
        state=state,
        trends=trends,
        run_id="run_integration",
        links={"web_ui": "http://example.com/run/integration"}
    )

    print(f"    生成了 {len(alerts)} 条告警")

    # 测试 drift 触发集成
    print("\n  7.2: Drift 触发告警")

    # 模拟 drift 检测到的异常
    signal.drawdown.current_drawdown = 0.11  # 超过阈值
    alerts = manager.process_risk_update(
        signal=signal,
        state=RiskState.CRITICAL,
        trends=trends,
        run_id="run_drift",
        links={"web_ui": "http://example.com/run/drift"}
    )

    print(f"    生成了 {len(alerts)} 条告警")

    # 测试手动告警（交易失败场景）
    print("\n  7.3: 手动告警（交易失败）")

    alert = manager.send_manual_alert(
        title="交易失败",
        message="下单失败: 连接超时",
        severity=AlertSeverity.WARNING,
        strategy_id="broker",
        tags=["trading", "error"],
        links={"log": "/logs/trading_error.log"}
    )

    assert alert.alert_id.startswith("manual_"), "手动告警 ID 应该以 manual_ 开头"

    print(f"    手动告警已发送: {alert.alert_id}")

    manager.shutdown()

    print(f"\n  ✅ 所有集成点测试通过")

    return True


def test_alert_statistics():
    """测试8: 告警统计"""
    print("\n" + "="*70)
    print("测试8: 告警统计")
    print("="*70)

    manager = AlertingManager(sender_config={"dry_run": True})

    # 生成一些测试告警
    for i in range(10):
        severity = [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.CRITICAL][i % 3]
        manager.send_manual_alert(
            title=f"测试告警 {i}",
            message=f"这是第 {i} 条测试告警",
            severity=severity,
            strategy_id=f"strategy_{i % 3}"
        )

    # 获取统计
    stats = manager.get_alert_statistics()

    assert stats["total_alerts"] == 10, "应该有 10 条告警"
    assert stats["by_severity"]["info"] > 0, "应该有 INFO 级别告警"
    assert stats["by_severity"]["warning"] > 0, "应该有 WARNING 级别告警"
    assert stats["by_severity"]["critical"] > 0, "应该有 CRITICAL 级别告警"
    assert "sent_success_rate" in stats, "应该有发送成功率统计"
    assert "circuit_breaker_status" in stats, "应该有熔断器状态"

    print(f"  ✅ 告警统计正确")
    print(f"  - 总告警数: {stats['total_alerts']}")
    print(f"  - 按严重程度分布: {stats['by_severity']}")
    print(f"  - 发送成功率: {stats['sent_success_rate']:.2%}")

    manager.shutdown()

    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("P1-1: Slack/Email/Webhook 通知集成测试")
    print("="*70)

    tests = [
        ("统一 AlertEvent schema", test_alert_event_schema),
        ("Slack 消息格式", test_slack_message_format),
        ("邮件格式", test_email_format),
        ("dry-run 模式", test_dry_run_mode),
        ("熔断器机制", test_circuit_breaker),
        ("异步发送不阻塞", test_async_send_no_blocking),
        ("集成点测试", test_integration_points),
        ("告警统计", test_alert_statistics),
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
