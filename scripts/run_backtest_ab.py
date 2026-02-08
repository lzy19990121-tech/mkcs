#!/usr/bin/env python3
"""
MKCS A/B 测试框架 (T3-1)

一键运行两个回测配置并生成对照报告
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

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ABTestRunner:
    """A/B 测试运行器"""

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        output_dir: str = "outputs/backtests",
        baseline_config: Dict = None,
        variant_config: Dict = None
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.baseline_config = baseline_config or {}
        self.variant_config = variant_config or {}

        # 默认配置
        self.default_config = {
            'capital': 100000,
            'max_risk': 0.02,
            'vote_threshold': 2,
            'min_strength': 0.001,
            'conflict_mode': 'HOLD'
        }

    def run_backtest(self, config: Dict, label: str) -> Dict:
        """运行单个回测"""
        # 合并配置
        full_config = {**self.default_config, **config}

        # 生成 run_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = "_".join(self.symbols[:3])
        run_id = f"ab_{label}_{symbols_str}_{timestamp}"

        # 构建命令
        cmd = [
            sys.executable,
            "scripts/run_backtest_strict.py",
            "--symbols", *self.symbols,
            "--start", self.start_date,
            "--end", self.end_date,
            "--capital", str(full_config['capital']),
            "--max-risk", str(full_config['max_risk']),
            "--vote-threshold", str(full_config['vote_threshold']),
            "--min-strength", str(full_config['min_strength']),
            "--conflict-mode", full_config['conflict_mode'],
            "--output-dir", str(self.output_dir),
            "--run-id", run_id
        ]

        logger.info(f"运行 {label} 回测...")
        logger.info(f"命令: {' '.join(cmd)}")

        # 运行回测
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root
        )

        if result.returncode != 0:
            logger.error(f"{label} 回测失败!")
            logger.error(result.stderr)
            return None

        # 读取结果
        run_dir = self.output_dir / run_id
        summary_path = run_dir / "summary.json"

        if not summary_path.exists():
            logger.error(f"{label} 回测未生成 summary.json")
            return None

        with open(summary_path) as f:
            summary = json.load(f)

        summary['run_dir'] = str(run_dir)
        summary['label'] = label

        logger.info(f"✅ {label} 回测完成")
        return summary

    def run_comparison(self) -> Dict:
        """运行 A/B 对比"""
        logger.info("=" * 60)
        logger.info("MKCS A/B 测试框架")
        logger.info("=" * 60)
        logger.info(f"标的: {', '.join(self.symbols)}")
        logger.info(f"区间: {self.start_date} ~ {self.end_date}")

        # 运行 baseline
        baseline = self.run_backtest(self.baseline_config, "baseline")
        if not baseline:
            return {'success': False, 'error': 'Baseline failed'}

        # 运行 variant
        variant = self.run_backtest(self.variant_config, "variant")
        if not variant:
            return {'success': False, 'error': 'Variant failed'}

        # 生成对照报告
        return self._generate_report(baseline, variant)

    def _generate_report(self, baseline: Dict, variant: Dict) -> Dict:
        """生成对照报告"""
        logger.info("\n" + "=" * 60)
        logger.info("生成 A/B 对照报告")
        logger.info("=" * 60)

        # 计算差异
        report = {
            'test_type': 'AB Comparison',
            'timestamp': datetime.now().isoformat(),
            'symbols': self.symbols,
            'period': f"{self.start_date} ~ {self.end_date}",
            'baseline': {
                'label': 'baseline',
                'config': self.baseline_config,
                'metrics': self._extract_metrics(baseline)
            },
            'variant': {
                'label': 'variant',
                'config': self.variant_config,
                'metrics': self._extract_metrics(variant)
            },
            'comparison': self._compare_metrics(baseline, variant)
        }

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"ab_comparison_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # 生成 Markdown 报告
        md_path = self._generate_markdown_report(report)

        logger.info(f"\n✅ 对照报告已保存:")
        logger.info(f"  - JSON: {report_path}")
        logger.info(f"  - Markdown: {md_path}")

        return {**report, 'success': True, 'report_path': str(report_path), 'md_path': str(md_path)}

    def _extract_metrics(self, result: Dict) -> Dict:
        """提取关键指标"""
        return {
            'total_return_pct': result.get('total_return_pct', 0),
            'max_drawdown_pct': result.get('max_drawdown_pct', 0),
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            'total_trades': result.get('total_trades', 0),
            'win_rate': result.get('win_rate', 0),
            'exit_reasons': result.get('exit_reasons', {})
        }

    def _compare_metrics(self, baseline: Dict, variant: Dict) -> Dict:
        """对比指标"""
        b = self._extract_metrics(baseline)
        v = self._extract_metrics(variant)

        return {
            'return_diff': v['total_return_pct'] - b['total_return_pct'],
            'return_improvement': (v['total_return_pct'] - b['total_return_pct']) / abs(b['total_return_pct']) * 100 if b['total_return_pct'] != 0 else 0,
            'drawdown_diff': v['max_drawdown_pct'] - b['max_drawdown_pct'],
            'drawdown_improvement': (b['max_drawdown_pct'] - v['max_drawdown_pct']) / b['max_drawdown_pct'] * 100 if b['max_drawdown_pct'] != 0 else 0,
            'sharpe_diff': v['sharpe_ratio'] - b['sharpe_ratio'],
            'trades_diff': v['total_trades'] - b['total_trades'],
            'winner': 'variant' if v['total_return_pct'] > b['total_return_pct'] else 'baseline'
        }

    def _generate_markdown_report(self, report: Dict) -> Path:
        """生成 Markdown 报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = self.output_dir / f"AB_COMPARISON_{timestamp}.md"

        b = report['baseline']['metrics']
        v = report['variant']['metrics']
        c = report['comparison']

        with open(md_path, 'w') as f:
            f.write("# MKCS A/B 测试对照报告\n\n")
            f.write(f"**生成时间**: {report['timestamp']}\n")
            f.write(f"**测试期间**: {report['period']}\n")
            f.write(f"**交易标的**: {', '.join(report['symbols'])}\n\n")

            f.write("## 📊 配置对比\n\n")
            f.write("| 参数 | Baseline | Variant |\n")
            f.write("|------|----------|--------|\n")

            for key in set(list(report['baseline']['config'].keys()) + list(report['variant']['config'].keys())):
                b_val = report['baseline']['config'].get(key, '-')
                v_val = report['variant']['config'].get(key, '-')
                f.write(f"| {key} | {b_val} | {v_val} |\n")

            f.write("\n## 📈 性能指标对比\n\n")
            f.write("| 指标 | Baseline | Variant | 差异 |\n")
            f.write("|------|----------|---------|------|\n")
            f.write(f"| 总收益 (%) | {b['total_return_pct']:.2f} | {v['total_return_pct']:.2f} | {c['return_diff']:+.2f} |\n")
            f.write(f"| 最大回撤 (%) | {b['max_drawdown_pct']:.2f} | {v['max_drawdown_pct']:.2f} | {c['drawdown_diff']:+.2f} |\n")
            f.write(f"| 夏普比率 | {b['sharpe_ratio']:.4f} | {v['sharpe_ratio']:.4f} | {c['sharpe_diff']:+.4f} |\n")
            f.write(f"| 总交易数 | {b['total_trades']} | {v['total_trades']} | {c['trades_diff']:+d} |\n")
            f.write(f"| 胜率 (%) | {b['win_rate']*100:.2f} | {v['win_rate']*100:.2f} | {(v['win_rate']-b['win_rate'])*100:+.2f} |\n")

            f.write("\n## 🎯 退出原因对比\n\n")
            f.write("### Baseline\n\n")
            for reason, count in b['exit_reasons'].items():
                f.write(f"- {reason}: {count}\n")

            f.write("\n### Variant\n\n")
            for reason, count in v['exit_reasons'].items():
                f.write(f"- {reason}: {count}\n")

            f.write("\n## 🏆 结论\n\n")
            if c['winner'] == 'variant':
                f.write(f"- **胜者**: Variant\n")
                f.write(f"- **收益提升**: {c['return_improvement']:+.2f}%\n")
                if c['drawdown_improvement'] != 0:
                    f.write(f"- **回撤改善**: {c['drawdown_improvement']:+.2f}%\n")
            else:
                f.write(f"- **胜者**: Baseline\n")
                f.write(f"- **收益降低**: {c['return_improvement']:+.2f}%\n")

        return md_path


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MKCS A/B 测试框架")
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'], help='交易标的')
    parser.add_argument('--start', default='2023-01-01', help='开始日期')
    parser.add_argument('--end', default='2024-12-31', help='结束日期')
    parser.add_argument('--output-dir', default='outputs/backtests', help='输出目录')

    # Baseline 配置
    parser.add_argument('--baseline-threshold', type=int, default=2, help='Baseline 投票阈值')
    parser.add_argument('--baseline-strength', type=float, default=0.001, help='Baseline 最小强度')

    # Variant 配置
    parser.add_argument('--variant-threshold', type=int, default=1, help='Variant 投票阈值')
    parser.add_argument('--variant-strength', type=float, default=0.001, help='Variant 最小强度')
    parser.add_argument('--variant-conflict', default='HOLD', choices=['HOLD', 'STRENGTH_DIFF'], help='Variant 冲突模式')

    args = parser.parse_args()

    # 创建配置
    baseline_config = {
        'vote_threshold': args.baseline_threshold,
        'min_strength': args.baseline_strength,
        'conflict_mode': 'HOLD'
    }

    variant_config = {
        'vote_threshold': args.variant_threshold,
        'min_strength': args.variant_strength,
        'conflict_mode': args.variant_conflict
    }

    # 运行 A/B 测试
    runner = ABTestRunner(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        baseline_config=baseline_config,
        variant_config=variant_config
    )

    result = runner.run_comparison()

    if result.get('success'):
        logger.info("\n🎉 A/B 测试完成!")
    else:
        logger.error(f"\n❌ A/B 测试失败: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
