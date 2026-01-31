"""
统一Replay输出Schema定义

定义回测结果的标准输出格式，确保可按时间切片和跨策略分析
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from decimal import Decimal
import pandas as pd
import json
from pathlib import Path


@dataclass
class TradeRecord:
    """成交记录"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # BUY/SELL
    price: Decimal
    quantity: int
    commission: Decimal
    signal_id: Optional[str] = None  # 关联信号ID


@dataclass
class StepRecord:
    """时间步记录"""
    timestamp: datetime
    strategy_id: str
    step_pnl: Decimal  # 当步盈亏
    equity: Decimal     # 当步权益
    run_id: str
    commit_hash: str
    config_hash: str
    cost_model: Dict[str, Any] = field(default_factory=dict)
    slippage: Dict[str, Any] = field(default_factory=dict)
    signal_state: Optional[Dict[str, Any]] = None


@dataclass
class ReplayOutput:
    """Replay输出标准格式

    必填字段在前，可选字段在后
    """
    # 必填字段（无默认值）
    run_id: str
    strategy_id: str
    strategy_name: str
    commit_hash: str
    config_hash: str
    start_date: date
    end_date: date
    initial_cash: Decimal
    final_equity: Decimal

    # 可选字段（有默认值）
    data_hash: Optional[str] = None
    steps: List[StepRecord] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_directory(cls, run_dir: Path) -> "ReplayOutput":
        """从run目录加载ReplayOutput

        Args:
            run_dir: 运行目录路径

        Returns:
            ReplayOutput对象
        """
        # 加载manifest
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # 加载summary
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")

        with open(summary_path) as f:
            summary = json.load(f)

        # 加载trades
        trades = []
        trades_path = run_dir / "trades.csv"
        if trades_path.exists():
            df = pd.read_csv(trades_path)
            for _, row in df.iterrows():
                trades.append(TradeRecord(
                    trade_id=row['trade_id'],
                    timestamp=pd.to_datetime(row['timestamp']),
                    symbol=row['symbol'],
                    side=row['side'],
                    price=Decimal(str(row['price'])),
                    quantity=int(row['quantity']),
                    commission=Decimal(str(row['commission']))
                ))

        # 加载equity curve
        equity_path = run_dir / "equity_curve.csv"
        if equity_path.exists():
            df = pd.read_csv(equity_path)
            df['date'] = pd.to_datetime(df['date'])
            df['timestamp'] = df['date']
        else:
            df = pd.DataFrame()

        # 构建steps
        steps = []
        if not df.empty:
            df = df.sort_values('timestamp')

            for _, row in df.iterrows():
                step = StepRecord(
                    timestamp=row['timestamp'],
                    strategy_id=manifest.get('strategy_name', 'unknown'),
                    step_pnl=Decimal(str(row['pnl'])),
                    equity=Decimal(str(row['equity'])),
                    run_id=manifest.get('experiment_id', 'unknown'),
                    commit_hash=manifest.get('git_commit', ''),
                    config_hash=manifest.get('config_hash', ''),
                    cost_model=summary.get('config', {}).get('cost_model', {}),
                    slippage=summary.get('config', {}).get('slippage', {})
                )
                steps.append(step)

        # 创建ReplayOutput
        replay = ReplayOutput(
            run_id=manifest.get('experiment_id', ''),
            strategy_id=manifest.get('strategy_name', 'unknown'),
            strategy_name=manifest.get('strategy_name', 'unknown'),
            commit_hash=manifest.get('git_commit', ''),
            config_hash=manifest.get('config_hash', ''),
            data_hash=summary.get('data_hash'),
            start_date=pd.to_datetime(summary['date_range']['start']).date(),
            end_date=pd.to_datetime(summary['date_range']['end']).date(),
            initial_cash=Decimal(str(summary['metrics']['initial_cash'])),
            final_equity=Decimal(str(summary['metrics']['final_equity'])),
            config=summary.get('config', {}),
            steps=steps,
            trades=trades
        )

        return replay

    def to_dataframe(self) -> pd.DataFrame:
        """转换为pandas DataFrame"""
        data = []
        for step in self.steps:
            data.append({
                'timestamp': step.timestamp,
                'strategy_id': step.strategy_id,
                'step_pnl': float(step.step_pnl),
                'equity': float(step.equity),
                'run_id': step.run_id,
                'commit_hash': step.commit_hash,
                'config_hash': step.config_hash
            })
        return pd.DataFrame(data)

    def slice_by_time(self, start: datetime, end: datetime) -> "ReplayOutput":
        """按时间切片"""
        filtered_steps = [
            s for s in self.steps
            if start <= s.timestamp <= end
        ]

        filtered_trades = [
            t for t in self.trades
            if start <= t.timestamp <= end
        ]

        # 创建新的ReplayOutput
        sliced = ReplayOutput(
            run_id=self.run_id + "_sliced",
            strategy_id=self.strategy_id,
            strategy_name=self.strategy_name,
            commit_hash=self.commit_hash,
            config_hash=self.config_hash,
            data_hash=self.data_hash,
            start_date=max(start.date(), self.start_date),
            end_date=min(end.date(), self.end_date),
            initial_cash=self.initial_cash,
            final_equity=self.final_equity,
            config=self.config,
            steps=filtered_steps,
            trades=filtered_trades
        )

        return sliced

    def get_equity_series(self) -> pd.Series:
        """获取权益序列"""
        df = self.to_dataframe()
        if df.empty:
            return pd.Series([], dtype=float)
        df = df.sort_values('timestamp')
        df['cumsum_pnl'] = df['step_pnl'].cumsum()
        df['equity_curve'] = float(self.initial_cash) + df['cumsum_pnl']
        return df.set_index('timestamp')['equity_curve']

    def get_returns_series(self) -> pd.Series:
        """获取收益率序列"""
        equity = self.get_equity_series()
        return equity.pct_change().dropna()


def load_replay_outputs(run_dir: str) -> List[ReplayOutput]:
    """加载目录下所有replay输出

    Args:
        run_dir: runs目录路径

    Returns:
        ReplayOutput列表
    """
    runs = []
    run_path = Path(run_dir)

    if not run_path.exists():
        return runs

    for exp_dir in run_path.iterdir():
        if not exp_dir.is_dir():
            continue

        manifest_path = exp_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue

        try:
            replay = ReplayOutput.from_directory(exp_dir)
            runs.append(replay)
        except Exception as e:
            print(f"Warning: Failed to load {exp_dir}: {e}")
            continue

    # 按时间排序
    runs.sort(key=lambda r: r.start_date)
    return runs


def create_standard_replay_output(
    run_dir: Path,
    steps_data: List[Dict],
    trades_data: List[Dict],
    manifest: Dict[str, Any],
    summary: Dict[str, Any]
) -> ReplayOutput:
    """创建标准格式的ReplayOutput并保存

    Args:
        run_dir: 运行目录
        steps_data: 步骤数据
        trades_data: 成交数据
        manifest: 清单数据
        summary: 摘要数据

    Returns:
        ReplayOutput对象
    """
    # 保存steps.csv
    steps_path = run_dir / "steps.csv"
    steps_df = pd.DataFrame(steps_data)
    steps_df.to_csv(steps_path, index=False)

    # 构建对象
    replay = ReplayOutput(
        run_id=manifest.get('experiment_id', ''),
        strategy_id=manifest.get('strategy_name', 'unknown'),
        strategy_name=manifest.get('strategy_name', 'unknown'),
        commit_hash=manifest.get('git_commit', ''),
        config_hash=manifest.get('config_hash', ''),
        data_hash=summary.get('data_hash'),
        start_date=datetime.strptime(summary['date_range']['start'], "%Y-%m-%d").date(),
        end_date=datetime.strptime(summary['date_range']['end'], "%Y-%m-%d").date(),
        initial_cash=Decimal(str(summary['metrics']['initial_cash'])),
        final_equity=Decimal(str(summary['metrics']['final_equity'])),
        config=summary.get('config', {}),
        steps=[],
        trades=[]
    )

    # 构建steps
    for data in steps_data:
        step = StepRecord(
            timestamp=pd.to_datetime(data['timestamp']),
            strategy_id=data.get('strategy_id', 'unknown'),
            step_pnl=Decimal(str(data['step_pnl'])),
            equity=Decimal(str(data['equity'])),
            run_id=manifest.get('experiment_id', ''),
            commit_hash=manifest.get('git_commit', ''),
            config_hash=manifest.get('config_hash', '')
        )
        replay.steps.append(step)

    # 构建trades
    for data in trades_data:
        trade = TradeRecord(
            trade_id=data['trade_id'],
            timestamp=pd.to_datetime(data['timestamp']),
            symbol=data['symbol'],
            side=data['side'],
            price=Decimal(str(data['price'])),
            quantity=int(data['quantity']),
            commission=Decimal(str(data['commission']))
        )
        replay.trades.append(trade)

    return replay


if __name__ == "__main__":
    """测试代码"""
    print("=== ReplaySchema 测试 ===\n")

    # 测试从目录加载
    runs = load_replay_outputs("runs")
    print(f"加载到 {len(runs)} 个回测结果")

    if runs:
        r = runs[0]
        print(f"\n示例: {r.run_id}")
        print(f"  策略: {r.strategy_id}")
        print(f"  日期: {r.start_date} 到 {r.end_date}")
        print(f"  初始资金: ${r.initial_cash:,.2f}")
        print(f"  最终权益: ${r.final_equity:,.2f}")
        print(f"  步骤数: {len(r.steps)}")
        print(f"  成交数: {len(r.trades)}")

        # 测试时间切片
        if r.steps:
            start = r.steps[0].timestamp
            end = r.steps[-1].timestamp
            mid = start + (end - start) / 2
            sliced = r.slice_by_time(start, mid)
            print(f"\n时间切片测试:")
            print(f"  原始步数: {len(r.steps)}")
            print(f"  切片步数: {len(sliced.steps)}")

    print("\n✓ 测试通过")
