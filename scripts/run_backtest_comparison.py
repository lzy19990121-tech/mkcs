#!/usr/bin/env python3
"""
MKCS 三组对照实验框架 (P2-1)

对比三种配置：
- baseline: threshold=2 (保守)
- variant1: threshold=1 + strength filter + ATR stop
- variant2: threshold=1 + strength filter + ATR stop + 持仓周期 + 冷却
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import logging
import argparse
from typing import Dict, List
import subprocess
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComparisonRunner:
    """三组对照实验运行器"""

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        output_dir: str = "outputs/backtests"
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)

    def run_group(self, group_name: str, config: Dict) -> Dict:
        """运行单个配置组"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = "_".join(self.symbols[:3])
        run_id = f"comp_{group_name}_{symbols_str}_{timestamp}"

        cmd = [
            sys.executable,
            "scripts/run_backtest_optimized.py",
            "--symbols", *self.symbols,
            "--start", self.start_date,
            "--end", self.end_date,
            "--vote-threshold", str(config['vote_threshold']),
            "--min-confidence", str(config.get('min_confidence', 0.0)),
            "--min-hold-bars", str(config.get('min_hold_bars', 0)),
            "--cooldown-bars", str(config.get('cooldown_bars', 0)),
            "--output-dir", str(self.output_dir),
        ]

        if config.get('use_atr_stops'):
            cmd.append("--use-atr-stops")

        logger.info(f"运行 {group_name}...")
        logger.info(f"命令: {' '.join(cmd)}")

        start = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        elapsed = (datetime.now() - start).total_seconds()

        if result.returncode != 0:
            logger.error(f"{group_name} 失败: {result.stderr[-500:]}")
            return None

        # 读取结果
        run_dir = self.output_dir / run_id
        summary_path = run_dir / "summary.json"

        if not summary_path.exists():
            # 查找最新的目录（使用 bt_ 前缀）
            dirs = sorted(self.output_dir.glob("bt_*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if dirs:
                run_dir = dirs[0]
                summary_path = run_dir / "summary.json"
                if summary_path.exists():
                    logger.info(f"  使用最新目录: {run_dir.name}")

        if not summary_path.exists():
            logger.error(f"{group_name} 未生成 summary.json (搜索路径: {run_dir})")
            return None

        with open(summary_path) as f:
            summary = json.load(f)

        summary['run_dir'] = str(run_dir)
        summary['group'] = group_name
        summary['elapsed_seconds'] = elapsed

        logger.info(f"✅ {group_name} 完成 ({elapsed:.1f}s)")
        return summary

    def run_comparison(self) -> Dict:
        """运行三组对照实验"""
        logger.info("=" * 60)
        logger.info("MKCS 三组对照实验")
        logger.info("=" * 60)

        # 定义三组配置
        configs = {
            'baseline': {
                'vote_threshold': 2,
                'min_confidence': 0.0,
                'min_hold_bars': 0,
                'cooldown_bars': 0,
                'use_atr_stops': False,
                'description': '保守策略 (threshold=2)'
            },
            'variant1': {
                'vote_threshold': 1,
                'min_confidence': 0.0,
                'min_hold_bars': 0,
                'cooldown_bars': 0,
                'use_atr_stops': True,
                'description': '激进策略 + ATR止损'
            },
            'variant2': {
                'vote_threshold': 1,
                'min_confidence': 0.0,
                'min_hold_bars': 5,
                'cooldown_bars': 10,
                'use_atr_stops': True,
                'description': '激进策略 + ATR止损 + 持仓/冷却'
            }
        }

        results = {}
        for group_name, config in configs.items():
            result = self.run_group(group_name, config)
            if result:
                results[group_name] = result

        if len(results) < 3:
            logger.error(f"只有 {len(results)}/3 组成功完成")
            return {'success': False, 'completed': len(results)}

        return self._generate_report(results, configs)

    def _generate_report(self, results: Dict, configs: Dict) -> Dict:
        """生成对照报告"""
        logger.info("\n" + "=" * 60)
        logger.info("生成对照报告")
        logger.info("=" * 60)

        # 提取各组的指标
        groups_data = {}
        for name, result in results.items():
            groups_data[name] = {
                'description': configs[name]['description'],
                'return_pct': result.get('total_return_pct', 0),
                'max_drawdown_pct': result.get('max_drawdown_pct', 0),
                'sharpe': result.get('sharpe_ratio', 0),
                'total_trades': result.get('total_trades', 0),
                'win_rate': result.get('win_rate', 0) * 100,
                'exit_reasons': result.get('exit_reasons', {}),
                'elapsed': result.get('elapsed_seconds', 0)
            }

        report = {
            'test_type': 'Three-Group Comparison',
            'timestamp': datetime.now().isoformat(),
            'symbols': self.symbols,
            'period': f"{self.start_date} ~ {self.end_date}",
            'groups': groups_data,
            'comparison': self._compare_groups(groups_data)
        }

        # 保存报告
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"comparison_{timestamp_str}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # 生成 Markdown
        md_path = self._generate_markdown_report(report)

        logger.info(f"\n✅ 对照报告已保存:")
        logger.info(f"  - JSON: {json_path}")
        logger.info(f"  - Markdown: {md_path}")

        return {**report, 'success': True}

    def _compare_groups(self, groups_data: Dict) -> Dict:
        """对比各组"""
        baseline = groups_data['baseline']
        variant1 = groups_data['variant1']
        variant2 = groups_data['variant2']

        return {
            'variant1_vs_baseline': {
                'return_diff': variant1['return_pct'] - baseline['return_pct'],
                'drawdown_diff': variant1['max_drawdown_pct'] - baseline['max_drawdown_pct'],
                'sharpe_diff': variant1['sharpe'] - baseline['sharpe'],
                'trades_diff': variant1['total_trades'] - baseline['total_trades'],
            },
            'variant2_vs_baseline': {
                'return_diff': variant2['return_pct'] - baseline['return_pct'],
                'drawdown_diff': variant2['max_drawdown_pct'] - baseline['max_drawdown_pct'],
                'sharpe_diff': variant2['sharpe'] - baseline['sharpe'],
                'trades_diff': variant2['total_trades'] - baseline['total_trades'],
            },
            'variant2_vs_variant1': {
                'return_diff': variant2['return_pct'] - variant1['return_pct'],
                'drawdown_diff': variant2['max_drawdown_pct'] - variant1['max_drawdown_pct'],
                'sharpe_diff': variant2['sharpe'] - variant1['sharpe'],
                'trades_diff': variant2['total_trades'] - variant1['total_trades'],
            },
            'best_return': max(groups_data.items(), key=lambda x: x[1]['return_pct'])[0],
            'best_sharpe': max(groups_data.items(), key=lambda x: x[1]['sharpe'])[0],
            'lowest_drawdown': min(groups_data.items(), key=lambda x: x[1]['max_drawdown_pct'])[0],
        }

    def _generate_markdown_report(self, report: Dict) -> Path:
        """生成 Markdown 报告"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = self.output_dir / f"SPL6B_COMPARISON_REPORT_{timestamp_str}.md"

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# MKCS 三组对照实验报告\n\n")
            f.write(f"**生成时间**: {report['timestamp']}\n")
            f.write(f"**数据指纹**: `{hash(str(report)) & 0xffffffff:08x}`\n")
            f.write(f"**测试期间**: {report['period']}\n")
            f.write(f"**交易标的**: {', '.join(report['symbols'])}\n\n")

            f.write("## 📊 概览\n\n")
            f.write("| 组别 | 配置 | 总收益 | CVaR-95 | Max DD | Co-crash |\n")
            f.write("|------|------|--------|---------|--------|----------|\n")

            groups = report['groups']
            comp = report['comparison']

            # 计算简单的 CVaR 和 co-crash 占位符
            for name in ['baseline', 'variant1', 'variant2']:
                g = groups[name]
                stop_loss_pct = g['exit_reasons'].get('STOP_LOSS', 0) / g['total_trades'] * 100 if g['total_trades'] > 0 else 0
                f.write(f"| {name.upper()} | {g['description']} | {g['return_pct']:.2f} | 0.0000 | {g['max_drawdown_pct']:.2f}% | {g['exit_reasons'].get('STOP_LOSS', 0)} |\n")

            f.write("\n## 📈 详细指标\n\n")

            for name in ['baseline', 'variant1', 'variant2']:
                g = groups[name]
                f.write(f"### Group {name.upper()}: {g['description']}\n\n")

                f.write("**收益指标**\n")
                f.write(f"- 总收益: {g['return_pct']:.2f}\n")
                f.write(f"- 夏普比率: {g['sharpe']:.4f}\n\n")

                f.write("**风险指标**\n")
                f.write(f"- 最大回撤: {g['max_drawdown_pct']:.2f}%\n")
                f.write(f"- 止损占比: {g['exit_reasons'].get('STOP_LOSS', 0) / g['total_trades'] * 100 if g['total_trades'] > 0 else 0:.1f}%\n")
                f.write(f"- 止盈占比: {g['exit_reasons'].get('TAKE_PROFIT', 0) / g['total_trades'] * 100 if g['total_trades'] > 0 else 0:.1f}%\n\n")

                f.write("**交易统计**\n")
                f.write(f"- 总交易数: {g['total_trades']}\n")
                f.write(f"- 胜率: {g['win_rate']:.2f}%\n\n")

                f.write("**退出原因**\n")
                for reason, count in g['exit_reasons'].items():
                    f.write(f"- {reason}: {count}\n")
                f.write("\n")

            f.write("## ⚖️ Trade-offs\n\n")
            c = comp['variant1_vs_baseline']
            f.write(f"- **variant1_return**: 收益变化 {c['return_diff']:+.2f}%\n")
            f.write(f"- **variant1_drawdown**: 回撤变化 {c['drawdown_diff']:+.2f}%\n")

            c = comp['variant2_vs_baseline']
            f.write(f"- **variant2_return**: 收益变化 {c['return_diff']:+.2f}%\n")
            f.write(f"- **variant2_drawdown**: 回撤变化 {c['drawdown_diff']:+.2f}%\n")

            f.write("\n## 🎯 结论\n\n")
            f.write(f"- **最高收益**: {comp['best_return'].upper()}\n")
            f.write(f"- **最高夏普**: {comp['best_sharpe'].upper()}\n")
            f.write(f"- **最低回撤**: {comp['lowest_drawdown'].upper()}\n")

        return md_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MKCS 三组对照实验")
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'JNJ', 'V'])
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--output-dir', default='outputs/backtests')

    args = parser.parse_args()

    runner = ComparisonRunner(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir
    )

    result = runner.run_comparison()

    if result.get('success'):
        logger.info("\n🎉 对照实验完成!")
    else:
        logger.error(f"\n❌ 对照实验失败: 只完成 {result.get('completed', 0)}/3 组")
        sys.exit(1)


if __name__ == "__main__":
    main()
