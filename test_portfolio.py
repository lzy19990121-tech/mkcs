"""
测试 SPL-4b 组合分析功能

构建2策略组合并运行协同风险分析
"""

from datetime import date
from pathlib import Path
import json

from analysis.portfolio import (
    PortfolioBuilder, PortfolioConfig,
    PortfolioWindowScanner, SynergyAnalyzer,
    PortfolioRiskReportGenerator
)
from analysis.replay_schema import load_replay_outputs


def main():
    """主测试流程"""
    print("=" * 70)
    print("SPL-4b 组合分析测试")
    print("=" * 70)

    # 1. 加载所有 replay
    print("\n1. 加载 replay 数据...")
    replays = load_replay_outputs('runs')
    print(f"   加载了 {len(replays)} 个 replay")

    # 筛选有足够数据的策略（超过20天）
    valid_replays = [
        r for r in replays
        if len(r.steps) > 20
    ]

    print(f"   有效策略（>20天）: {len(valid_replays)}")
    for r in valid_replays:
        print(f"     - {r.run_id}: {r.strategy_id} ({len(r.steps)} steps)")

    if len(valid_replays) < 2:
        print("\n⚠️  需要至少2个有效策略来构建组合")
        return

    # 2. 手动构建组合（因为 strategy_id 都是 "unknown"）
    print("\n2. 手动构建组合...")

    # 使用前2个有效策略
    replay_1 = valid_replays[0]
    replay_2 = valid_replays[1]

    print(f"   策略1: {replay_1.run_id} ({replay_1.start_date} ~ {replay_1.end_date})")
    print(f"   策略2: {replay_2.run_id} ({replay_2.start_date} ~ {replay_2.end_date})")

    # 确定时间范围（使用两个策略的交集）
    start_date = max(replay_1.start_date, replay_2.start_date)
    end_date = min(replay_1.end_date, replay_2.end_date)

    print(f"   时间范围: {start_date} ~ {end_date}")

    # 手动对齐数据
    from analysis.portfolio import Portfolio
    from datetime import datetime
    import pandas as pd

    # 转换为 DataFrame
    df_1 = replay_1.to_dataframe()
    df_2 = replay_2.to_dataframe()

    # 过滤时间范围
    df_1 = df_1[(df_1['timestamp'] >= pd.Timestamp(start_date)) &
                (df_1['timestamp'] <= pd.Timestamp(end_date))]
    df_2 = df_2[(df_2['timestamp'] >= pd.Timestamp(start_date)) &
                (df_2['timestamp'] <= pd.Timestamp(end_date))]

    # 找到共同的时间点
    common_times = pd.Series(list(set(df_1['timestamp']) & set(df_2['timestamp'])))
    common_times = common_times.sort_values()

    print(f"   共同时间点数: {len(common_times)}")

    if len(common_times) < 5:
        print("\n⚠️  共同时间点太少，无法构建组合")
        return

    # 对齐数据
    df_1_aligned = df_1.set_index('timestamp').loc[common_times]
    df_2_aligned = df_2.set_index('timestamp').loc[common_times]

    # 计算收益
    returns_1 = df_1_aligned['step_pnl'].values
    returns_2 = df_2_aligned['step_pnl'].values

    # 计算权益曲线
    initial_value = 100000.0
    equity_1 = initial_value + pd.Series(returns_1).cumsum().values
    equity_2 = initial_value + pd.Series(returns_2).cumsum().values

    # 计算组合权益 (60% 策略1, 40% 策略2)
    combined_equity = 0.6 * equity_1 + 0.4 * equity_2
    combined_returns = 0.6 * returns_1 + 0.4 * returns_2

    # 创建 Portfolio 对象
    portfolio = Portfolio(
        portfolio_id=f"portfolio_{replay_1.run_id}_{replay_2.run_id}",
        config=PortfolioConfig(
            strategy_ids=[replay_1.run_id, replay_2.run_id],
            weights={replay_1.run_id: 0.6, replay_2.run_id: 0.4},
            start_date=start_date,
            end_date=end_date,
            alignment_method="inner"
        ),
        timestamps=common_times.tolist(),
        combined_equity=combined_equity.tolist(),
        combined_returns=combined_returns.tolist(),
        strategy_equities={
            replay_1.run_id: equity_1.tolist(),
            replay_2.run_id: equity_2.tolist()
        },
        strategy_returns={
            replay_1.run_id: returns_1.tolist(),
            replay_2.run_id: returns_2.tolist()
        },
        initial_value=initial_value,
        final_value=combined_equity[-1],
        total_return=(combined_equity[-1] - initial_value) / initial_value
    )

    print(f"   ✓ 组合构建成功")
    print(f"     组合ID: {portfolio.portfolio_id}")
    print(f"     初始价值: {portfolio.initial_value:,.2f}")
    print(f"     最终价值: {portfolio.final_value:,.2f}")
    print(f"     总收益: {portfolio.total_return*100:.2f}%")
    print(f"     时间点数: {len(portfolio.timestamps)}")

    # 4. 扫描最坏窗口
    print("\n4. 扫描组合最坏窗口...")
    worst_windows = []

    try:
        scanner = PortfolioWindowScanner()
        worst_windows = scanner.find_worst_portfolio_windows(
            portfolio,
            window_lengths=["20d"],
            top_k=3
        )
        print(f"   ✓ 找到 {len(worst_windows)} 个最坏窗口")

        for i, win in enumerate(worst_windows, 1):
            print(f"\n     窗口 #{i}: {win.window_id}")
            print(f"       时间: {win.start_date} ~ {win.end_date}")
            print(f"       收益: {win.total_return*100:.2f}%")
            print(f"       最大回撤: {win.max_drawdown*100:.2f}%")
    except Exception as e:
        print(f"   ✗ 窗口扫描失败: {e}")
        worst_windows = []

    # 5. 协同风险分析
    print("\n5. 协同风险分析...")
    analyzer = SynergyAnalyzer()

    try:
        synergy_report = analyzer.generate_synergy_report(
            portfolio,
            worst_windows,
            risk_budget=-0.10  # -10% 风险预算
        )
        print(f"   ✓ 协同分析完成")
        print(f"\n     协同风险统计:")
        print(f"       不安全组合: {len(synergy_report.unsafe_combinations)}")
        print(f"       尾部损失事件: {len(synergy_report.simultaneous_tail_losses)}")
        print(f"       相关性尖峰: {len(synergy_report.correlation_spikes)}")
        print(f"       风险预算违规: {len(synergy_report.risk_budget_breaches)}")

        # 显示策略贡献
        if synergy_report.strategy_contributions:
            print(f"\n     策略贡献分析:")
            for strat_id, contrib in synergy_report.strategy_contributions.items():
                print(f"       {strat_id}:")
                print(f"         最坏贡献: {contrib*100:.2f}%")
    except Exception as e:
        print(f"   ✗ 协同分析失败: {e}")
        import traceback
        traceback.print_exc()
        synergy_report = None

    # 6. 生成报告
    print("\n6. 生成组合风险报告...")
    output_dir = Path("reports/portfolio")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "test_portfolio_report.md"

    try:
        generator = PortfolioRiskReportGenerator()
        report = generator.generate_report(
            portfolio,
            worst_windows,
            synergy_report,
            output_path=output_path
        )
        print(f"   ✓ 报告已生成: {output_path}")
    except Exception as e:
        print(f"   ✗ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()

    # 7. 保存组合数据
    print("\n7. 保存组合数据...")
    try:
        # 保存为 JSON
        portfolio_data = {
            "portfolio_id": portfolio.portfolio_id,
            "config": {
                "strategy_ids": portfolio.config.strategy_ids,
                "weights": portfolio.config.weights,
                "start_date": portfolio.config.start_date.isoformat(),
                "end_date": portfolio.config.end_date.isoformat(),
            },
            "metrics": {
                "initial_value": portfolio.initial_value,
                "final_value": portfolio.final_value,
                "total_return": portfolio.total_return
            },
            "timestamps": [ts.isoformat() for ts in portfolio.timestamps[:10]],  # 只保存前10个
            "combined_equity": portfolio.combined_equity[:10],
            "combined_returns": portfolio.combined_returns[:10]
        }

        json_path = output_dir / "test_portfolio_data.json"
        with open(json_path, 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        print(f"   ✓ 组合数据已保存: {json_path}")

    except Exception as e:
        print(f"   ✗ 保存失败: {e}")

    # 8. 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"组合ID: {portfolio.portfolio_id}")
    print(f"策略数: {len(portfolio.config.strategy_ids)}")
    print(f"总收益: {portfolio.total_return*100:.2f}%")
    print(f"最坏窗口数: {len(worst_windows)}")

    if synergy_report:
        print(f"协同风险:")
        print(f"  不安全组合: {len(synergy_report.unsafe_combinations)}")
        print(f"  尾部损失事件: {len(synergy_report.simultaneous_tail_losses)}")
        print(f"  相关性尖峰: {len(synergy_report.correlation_spikes)}")
        print(f"  风险预算违规: {len(synergy_report.risk_budget_breaches)}")

    print("\n✅ SPL-4b 组合分析测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
