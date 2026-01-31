#!/usr/bin/env python
"""
策略对比脚本

对同一数据集、同一风控，跑多个策略/参数组合
输出对比报告和表格
"""

import sys
import os
import csv
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BacktestConfig
from agent.runner import run_backtest_with_config
from skills.strategy.moving_average import MAStrategy
from skills.strategy.breakout import BreakoutStrategy
from skills.risk.basic_risk import BasicRiskManager
from skills.market_data.mock_source import MockMarketSource
from broker.paper import PaperBroker
from agent.runner import TradingAgent
from events.event_log import EventLogger


def calculate_max_drawdown(equity_curve: List[Dict]) -> float:
    """计算最大回撤"""
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]['equity']
    max_dd = 0.0

    for point in equity_curve:
        equity = point['equity']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return max_dd


def calculate_win_rate(trades: List[Dict]) -> float:
    """计算胜率"""
    if not trades:
        return 0.0

    # 简化计算：买入卖出配对
    buy_trades = [t for t in trades if t['side'] == 'BUY']
    sell_trades = [t for t in trades if t['side'] == 'SELL']

    if not buy_trades or not sell_trades:
        return 0.0

    # 匹配买卖
    wins = 0
    total = 0

    for sell in sell_trades:
        # 找到最近的买入
        for buy in reversed(buy_trades):
            if buy['timestamp'] < sell['timestamp']:
                pnl = (sell['price'] - buy['price']) * buy['quantity']
                if pnl > 0:
                    wins += 1
                total += 1
                buy_trades.remove(buy)
                break

    return wins / total if total > 0 else 0.0


def run_strategy_comparison(
    strategies: List[Dict[str, Any]],
    symbols: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = "runs/comparison"
) -> List[Dict[str, Any]]:
    """运行策略对比

    Args:
        strategies: 策略列表 [{"name": "ma_fast", "strategy": MAStrategy(...)}, ...]
        symbols: 交易标的
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录

    Returns:
        对比结果列表
    """
    results = []
    comparison_dir = Path(output_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # 共享配置
    base_config = {
        "data_source": "mock",
        "seed": 42,
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "commission_per_share": 0.01,
        "slippage_bps": 2,
        "initial_cash": 100000.0
    }

    # 共享风控
    risk_manager = BasicRiskManager(
        max_position_ratio=0.2,
        max_positions=5,
        blacklist=set(),
        max_daily_loss_ratio=0.05
    )

    # 运行每个策略
    for strategy_config in strategies:
        strategy_name = strategy_config["name"]
        strategy = strategy_config["strategy"]

        print(f"\n运行策略: {strategy_name}")

        # 创建配置
        config = BacktestConfig(
            **base_config,
            strategy_name=strategy_name,
            strategy_params=getattr(strategy, '__dict__', {})
        )

        # 创建agent
        data_source = MockMarketSource(seed=base_config["seed"])
        broker = PaperBroker(
            initial_cash=base_config["initial_cash"],
            commission_per_share=base_config["commission_per_share"],
            slippage_bps=base_config["slippage_bps"]
        )
        event_logger = EventLogger(log_dir=f"logs/{strategy_name}")

        agent = TradingAgent(
            data_source=data_source,
            strategy=strategy,
            risk_manager=risk_manager,
            broker=broker,
            event_logger=event_logger,
            config=config
        )

        # 运行回测
        from datetime import datetime
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        agent.run_replay_backtest(
            start=start,
            end=end,
            symbols=symbols,
            interval="1d",
            output_dir=str(comparison_dir / strategy_name),
            verbose=False,
            use_manifest=False  # 使用固定目录结构便于对比
        )

        # 读取结果
        summary_path = comparison_dir / strategy_name / "summary.json"
        with open(summary_path) as f:
            summary = json.load(f)

        # 读取权益曲线
        equity_curve = []
        equity_path = comparison_dir / strategy_name / "equity_curve.csv"
        with open(equity_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['equity'] = float(row['equity'])
                row['pnl'] = float(row['pnl'])
                equity_curve.append(row)

        # 读取交易记录
        trades = []
        trades_path = comparison_dir / strategy_name / "trades.csv"
        with open(trades_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['price'] = float(row['price'])
                row['quantity'] = int(row['quantity'])
                trades.append(row)

        # 读取风险拒绝记录
        risk_rejects = []
        rejects_path = comparison_dir / strategy_name / "risk_rejects.csv"
        if rejects_path.exists():
            with open(rejects_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    risk_rejects.append(row)

        # 计算指标
        max_dd = calculate_max_drawdown(equity_curve)
        win_rate = calculate_win_rate(trades)

        # 计算平均持仓时间
        avg_hold = 0.0
        if trades:
            hold_times = []
            buy_trades = [t for t in trades if t['side'] == 'BUY']
            sell_trades = [t for t in trades if t['side'] == 'SELL']

            for sell in sell_trades:
                sell_time = datetime.fromisoformat(sell['timestamp'])
                for buy in reversed(buy_trades):
                    buy_time = datetime.fromisoformat(buy['timestamp'])
                    if buy_time < sell_time:
                        hold_days = (sell_time - buy_time).days
                        hold_times.append(hold_days)
                        buy_trades.remove(buy)
                        break

            if hold_times:
                avg_hold = sum(hold_times) / len(hold_times)

        # 汇总结果
        result = {
            "strategy": strategy_name,
            "experiment_id": summary.get("config_hash", "unknown")[:16],
            "total_return": summary["metrics"]["total_return"],
            "max_drawdown": max_dd,
            "trade_count": summary["metrics"]["trade_count"],
            "win_rate": win_rate,
            "avg_hold_days": avg_hold,
            "risk_reject_count": len(risk_rejects),
            "final_equity": summary["metrics"]["final_equity"],
            "run_path": str(comparison_dir / strategy_name)
        }

        results.append(result)

        print(f"  总收益: {result['total_return']*100:.2f}%")
        print(f"  最大回撤: {result['max_drawdown']*100:.2f}%")
        print(f"  交易次数: {result['trade_count']}")
        print(f"  胜率: {result['win_rate']*100:.1f}%")

    return results


def generate_comparison_report(results: List[Dict], output_dir: str):
    """生成对比报告"""
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 生成 Markdown 报告
    md_path = report_dir / "compare_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 策略对比报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 表格
        f.write("## 策略对比表\n\n")
        f.write("| 策略 | 总收益 | 最大回撤 | 交易次数 | 胜率 | 平均持仓(天) | 风控拒绝 | 最终权益 |\n")
        f.write("|------|--------|----------|----------|------|-------------|----------|----------|\n")

        for r in results:
            f.write(f"| {r['strategy']} | ")
            f.write(f"{r['total_return']*100:.2f}% | ")
            f.write(f"{r['max_drawdown']*100:.2f}% | ")
            f.write(f"{r['trade_count']} | ")
            f.write(f"{r['win_rate']*100:.1f}% | ")
            f.write(f"{r['avg_hold_days']:.1f} | ")
            f.write(f"{r['risk_reject_count']} | ")
            f.write(f"${r['final_equity']:,.2f} |\n")

        # 详细结果
        f.write("\n## 详细结果\n\n")
        for r in results:
            f.write(f"### {r['strategy']}\n\n")
            f.write(f"- **实验ID**: `{r['experiment_id']}`\n")
            f.write(f"- **总收益**: {r['total_return']*100:.2f}%\n")
            f.write(f"- **最大回撤**: {r['max_drawdown']*100:.2f}%\n")
            f.write(f"- **交易次数**: {r['trade_count']}\n")
            f.write(f"- **胜率**: {r['win_rate']*100:.1f}%\n")
            f.write(f"- **平均持仓**: {r['avg_hold_days']:.1f}天\n")
            f.write(f"- **风控拒绝**: {r['risk_reject_count']}\n")
            f.write(f"- **结果路径**: `{r['run_path']}`\n\n")

    # 生成 CSV 表格
    csv_path = report_dir / "compare_table.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['strategy', 'experiment_id', 'total_return', 'max_drawdown', 'trade_count',
                     'win_rate', 'avg_hold_days', 'risk_reject_count', 'final_equity', 'run_path']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✓ 对比报告已生成:")
    print(f"  Markdown: {md_path}")
    print(f"  CSV: {csv_path}")


def main():
    """主函数"""
    print("="*60)
    print("策略对比框架")
    print("="*60)

    # 定义策略对比
    strategies = [
        {
            "name": "ma_5_20",
            "strategy": MAStrategy(fast_period=5, slow_period=20)
        },
        {
            "name": "ma_3_10",
            "strategy": MAStrategy(fast_period=3, slow_period=10)
        },
        {
            "name": "breakout_20",
            "strategy": BreakoutStrategy(period=20, threshold=0.01)
        },
        {
            "name": "breakout_10",
            "strategy": BreakoutStrategy(period=10, threshold=0.01)
        }
    ]

    # 运行对比
    results = run_strategy_comparison(
        strategies=strategies,
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2024-01-01",
        end_date="2024-06-30",
        output_dir="runs/comparison"
    )

    # 生成报告
    generate_comparison_report(results, "runs/comparison")

    print("\n" + "="*60)
    print("✓ 策略对比完成")
    print("="*60)


if __name__ == "__main__":
    main()
