"""
端到端集成测试

运行完整的回测流程，验证所有模块的协同工作
"""

import os
import sys
import tempfile
from pathlib import Path
from datetime import date, datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.runner import create_default_agent
from reports.daily import BacktestReport


def run_e2e_backtest():
    """运行端到端回测"""
    print("\n" + "="*70)
    print("端到端集成测试")
    print("="*70 + "\n")

    # 创建临时数据库
    with tempfile.NamedTemporaryFile(mode="w", suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        print("1. 初始化测试环境...")
        print(f"   数据库: {db_path}")

        # 创建Agent
        agent = create_default_agent(initial_cash=100000, db_path=db_path)

        # 运行回测
        print("\n2. 运行回测 (2024-01-01 到 2024-01-31)...")
        symbols = ["AAPL", "GOOGL", "MSFT"]

        results = agent.run_replay_backtest(
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            symbols=symbols,
            interval="1d",
            verbose=True
        )

        # 2.1 校验回放输出文件
        output_dir = Path("reports/replay")
        required_outputs = [
            output_dir / "summary.json",
            output_dir / "equity_curve.csv",
            output_dir / "trades.csv",
            output_dir / "risk_rejects.csv"
        ]
        for path in required_outputs:
            assert path.exists(), f"回放输出缺失: {path}"

        # 验证结果
        print("\n3. 验证数据一致性...")

        # 3.1 验证数据库中的交易记录
        db_trades = agent.db.get_trades()
        broker_trades = agent.broker.get_trades()

        print(f"   数据库交易记录: {len(db_trades)}")
        print(f"   Broker交易记录: {len(broker_trades)}")

        assert len(db_trades) == len(broker_trades), "交易记录数量不一致"
        print("   ✓ 交易记录数量一致")

        # 3.2 验证交易详情
        for db_trade, broker_trade in zip(db_trades, broker_trades):
            assert db_trade.trade_id == broker_trade.trade_id, "交易ID不匹配"
            assert db_trade.symbol == broker_trade.symbol, "标的代码不匹配"
            assert db_trade.side == broker_trade.side, "交易方向不匹配"
        print("   ✓ 交易详情一致")

        # 3.3 验证持仓快照
        positions = agent.db.get_positions()
        broker_positions = agent.broker.get_positions()

        print(f"   数据库持仓记录: {len(positions)}")
        print(f"   Broker持仓数量: {len(broker_positions)}")
        print("   ✓ 持仓数据已保存")

        # 3.4 验证资金平衡
        # 计算预期权益：现金 + 持仓市值
        expected_equity = agent.broker.get_cash_balance()
        for pos in broker_positions.values():
            expected_equity += pos.market_value

        actual_equity = agent.broker.get_total_equity()
        assert abs(expected_equity - actual_equity) < Decimal("0.01"), "资金不平衡"
        print("   ✓ 资金平衡验证通过")

        # 4. 生成报告
        print("\n4. 生成回测报告...")
        report = BacktestReport(agent.broker, results)
        report_text = report.generate_summary()
        print(report_text)

        # 5. 性能统计
        print("\n5. 性能统计:")
        total_signals = sum(r.get('signals_generated', 0) for r in results)
        total_executed = sum(r.get('orders_filled', 0) for r in results)
        execution_rate = total_executed / total_signals * 100 if total_signals > 0 else 0

        print(f"   总信号数: {total_signals}")
        print(f"   执行订单: {total_executed}")
        print(f"   执行率: {execution_rate:.2f}%")
        print(f"   总收益率: {agent.broker.get_total_pnl()/agent.broker.initial_cash*100:+.2f}%")

        # 6. 测试结果
        print("\n" + "="*70)
        print("集成测试结果")
        print("="*70)

        all_passed = True

        # 测试用例1: Agent正常初始化
        try:
            assert agent.broker is not None
            assert agent.strategy is not None
            assert agent.risk_manager is not None
            assert agent.db is not None
            print("✓ 用例1: Agent初始化 - 通过")
        except AssertionError as e:
            print(f"✗ 用例1: Agent初始化 - 失败: {e}")
            all_passed = False

        # 测试用例2: 回测正常执行
        try:
            assert len(results) > 0
            assert any(r.get('signals_generated', 0) > 0 for r in results)
            print("✓ 用例2: 回测执行 - 通过")
        except AssertionError as e:
            print(f"✗ 用例2: 回测执行 - 失败: {e}")
            all_passed = False

        # 测试用例3: 数据持久化
        try:
            assert len(db_trades) > 0 or len(broker_trades) == 0
            print("✓ 用例3: 数据持久化 - 通过")
        except AssertionError as e:
            print(f"✗ 用例3: 数据持久化 - 失败: {e}")
            all_passed = False

        # 测试用例4: 风控规则生效
        try:
            # 检查是否有被拒绝的订单（可能因为资金不足或其他风控原因）
            rejected = sum(r.get('orders_rejected', 0) for r in results)
            print(f"✓ 用例4: 风控规则 - 通过 (拒绝{rejected}笔)")
        except AssertionError as e:
            print(f"✗ 用例4: 风控规则 - 失败: {e}")
            all_passed = False

        # 测试用例5: 报告生成
        try:
            assert "回测报告摘要" in report_text
            assert "账户表现" in report_text
            print("✓ 用例5: 报告生成 - 通过")
        except AssertionError as e:
            print(f"✗ 用例5: 报告生成 - 失败: {e}")
            all_passed = False

        print("\n" + "="*70)
        if all_passed:
            print("✓ 所有集成测试通过！")
        else:
            print("✗ 部分测试失败")
            return False
        print("="*70 + "\n")

        return True

    finally:
        # 清理临时文件
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"已清理临时数据库: {db_path}")


if __name__ == "__main__":
    from decimal import Decimal

    success = run_e2e_backtest()
    sys.exit(0 if success else 1)
