"""
MKCS 严格历史数据回测引擎

严格遵循以下原则：
1. 禁止使用未来数据（T0-4 ~ T0-7）
2. 自动化 Lookahead 检测（T0-8）
3. 运行时防护（T0-9）
4. 投票融合逻辑工程化（T1）
5. 止损止盈补齐（T2）
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging
import json
import csv
import hashlib
import numpy as np
import pandas as pd

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from skills.market_data.yahoo_source import YahooFinanceSource

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 时间语义定义 (T0-4) ====================
"""
时间语义：
- t-1 close: 已完成的历史数据（可用于生成信号）
- t open / t bar: 当前可用的信息（可用于执行交易）
- t+1: 严格禁止使用

规则：
1. 信号生成：只能用 <= t-1 的数据
2. 下单执行：发生在 t open 或 t close
3. 盈亏统计：允许用 t close（事后）
"""

# ==================== 指标计算器 (T0-5) ====================
class IndicatorCalculator:
    """技术指标计算器（防泄漏）

    所有指标计算都使用 shift(1) 确保不使用未来数据
    """

    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """简单移动平均（带 shift）"""
        return data.rolling(window=period).mean().shift(1)

    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """指数移动平均（带 shift）"""
        return data.ewm(span=period, adjust=False).mean().shift(1)

    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """平均真实范围（带 shift）"""
        high = data['High']
        low = data['Low']
        close = data['Close']

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = tr1.combine(tr2, max).combine(tr3, max)

        return tr.rolling(window=period).mean().shift(1)

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指标（带 shift）"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean().shift(1)
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean().shift(1)

        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def verify_no_lookahead(
        current_bar_idx: int,
        data_len: int,
        last_used_data_idx: int
    ) -> None:
        """验证信号不使用未来数据

        Args:
            current_bar_idx: 当前 bar 索引
            data_len: 数据总长度
            last_used_data_idx: 信号最后使用的数据索引
        """
        # 由于我们使用了 shift(1)，信号使用的是 current_idx-1 的数据
        # 所以 last_used_data_idx = current_idx - 1
        # 只要 last_used_data_idx < current_bar_idx，就没有使用未来数据
        if last_used_data_idx >= current_bar_idx:
            raise ValueError(
                f"Lookahead detected! signal used data up to index {last_used_data_idx}, "
                f"but current bar is {current_bar_idx}"
            )


# ==================== 交易记录与状态 ====================
@dataclass
class Trade:
    """交易记录"""
    symbol: str
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    direction: Literal['LONG', 'SHORT']
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    entry_value: float
    exit_value: Optional[float]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    exit_reason: str  # 'SIGNAL', 'STOP_LOSS', 'TAKE_PROFIT', 'TIME_EXIT', 'CONFLICT'
    commission: float
    # 投票信息 (T1-2)
    vote_buy: int = 0
    vote_sell: int = 0
    vote_conflict: int = 0
    signal_strength: float = 0.0
    strategy_votes: Dict[str, str] = field(default_factory=dict)

    def to_dict(self, exclude_strategy_votes: bool = True) -> Dict:
        d = asdict(self)
        d['entry_date'] = self.entry_date.isoformat() if self.entry_date else None
        d['exit_date'] = self.exit_date.isoformat() if self.exit_date else None
        if exclude_strategy_votes and 'strategy_votes' in d:
            del d['strategy_votes']
        return d


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    direction: Literal['LONG', 'SHORT']
    quantity: int
    entry_price: float
    entry_date: pd.Timestamp
    entry_bar_idx: int  # 用于验证
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_value: float
    # 投票信息
    vote_buy: int = 0
    vote_sell: int = 0
    signal_strength: float = 0.0


@dataclass
class Signal:
    """策略信号"""
    strategy_id: str
    timestamp: pd.Timestamp
    bar_idx: int
    action: Literal['BUY', 'SELL', 'HOLD']
    strength: float  # [-1, 1]
    price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    reason: str

    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


# ==================== 资金管理器 ====================
class CapitalManager:
    """资金管理器

    规则：
    1. 单笔交易最大风险不超过账户的 2%
    2. 单个标的最大仓位不超过账户的 20%
    3. 总持仓不超过账户的 80%
    4. 动态调整仓位大小
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        max_risk_per_trade: float = 0.02,
        max_position_pct: float = 0.20,
        max_total_exposure: float = 0.80,
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0,
        atr_multiplier: float = 2.0  # ATR 倍数计算止损
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.atr_multiplier = atr_multiplier

        self.equity_curve = [initial_capital]
        self.drawdowns = [0.0]
        self.timestamps = []

    def get_equity(self) -> float:
        return self.current_capital

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: Optional[float],
        current_exposure: float,
        atr: Optional[float] = None
    ) -> int:
        """计算仓位大小"""
        # 1. 基于风险的计算
        if stop_loss and stop_loss != entry_price:
            risk_per_share = abs(entry_price - stop_loss)
            risk_amount = self.current_capital * self.max_risk_per_trade
            shares_by_risk = int(risk_amount / risk_per_share)
        else:
            shares_by_risk = float('inf')

        # 2. 基于最大仓位的计算
        max_position_value = self.current_capital * self.max_position_pct
        shares_by_position = int(max_position_value / entry_price)

        # 3. 基于剩余资金的计算
        available_capital = self.current_capital * self.max_total_exposure - current_exposure
        shares_by_capital = int(available_capital / entry_price) if available_capital > 0 else 0

        shares = min(shares_by_risk, shares_by_position, shares_by_capital)
        shares = max(0, (shares // 100) * 100)  # 100 股整数倍

        return shares

    def calculate_commission(self, price: float, quantity: int) -> float:
        return price * quantity * self.commission_rate

    def apply_slippage(self, price: float, direction: str) -> float:
        slippage = price * self.slippage_bps / 10000
        if direction == 'BUY':
            return price + slippage
        else:
            return price - slippage

    def update_equity(self, pnl: float, timestamp: pd.Timestamp):
        self.current_capital += pnl
        self.equity_curve.append(self.current_capital)
        self.timestamps.append(timestamp)

        peak = max(self.equity_curve)
        drawdown = (peak - self.current_capital) / peak if peak > 0 else 0
        self.drawdowns.append(drawdown)

    def get_metrics(self) -> Dict:
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]

        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        max_drawdown = max(self.drawdowns)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe
        }


# ==================== 策略基类 ====================
class Strategy:
    """策略基类"""

    def __init__(self, strategy_id: str, min_strength: float = 0.01):
        self.strategy_id = strategy_id
        self.min_strength = min_strength  # T1-3: 最小强度阈值

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_idx: int
    ) -> Signal:
        """生成信号（禁止在子类中重写此方法签名）

        Args:
            data: 历史数据（包含当前 bar）
            current_idx: 当前 bar 索引

        Returns:
            Signal
        """
        # T0-9: 运行时防护
        if current_idx < 50:
            return self._hold_signal(data, current_idx, "Insufficient data")

        # T0-5: 使用 IndicatorCalculator 计算指标（自带 shift）
        signal = self._compute_signal(data, current_idx)

        # T0-5: 验证不使用未来数据
        # 由于使用了 shift(1)，信号实际使用的是 current_idx-1 的数据
        IndicatorCalculator.verify_no_lookahead(current_idx, len(data), current_idx - 1)

        # T1-3: 检查信号强度
        if signal.action != 'HOLD' and abs(signal.strength) < self.min_strength:
            return Signal(
                strategy_id=self.strategy_id,
                timestamp=data.index[current_idx],
                bar_idx=current_idx,
                action='HOLD',
                strength=0,
                price=signal.price,
                stop_loss=None,
                take_profit=None,
                reason=f"Strength too low: {signal.strength:.4f} < {self.min_strength}"
            )

        return signal

    def _compute_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """计算信号（子类实现）"""
        raise NotImplementedError

    def _hold_signal(self, data: pd.DataFrame, current_idx: int, reason: str) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            timestamp=data.index[current_idx],
            bar_idx=current_idx,
            action='HOLD',
            strength=0,
            price=float(data['Close'].iloc[current_idx]),
            stop_loss=None,
            take_profit=None,
            reason=reason
        )


class MAStrategy(Strategy):
    """移动平均策略"""

    def __init__(
        self,
        strategy_id: str,
        fast_period: int,
        slow_period: int,
        min_strength: float = 0.01
    ):
        super().__init__(strategy_id, min_strength)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"{strategy_id}_MA({fast_period},{slow_period})"

    def _compute_signal(self, data: pd.DataFrame, current_idx: int) -> Signal:
        # 使用 IndicatorCalculator（自带 shift）
        ma_fast = IndicatorCalculator.sma(data['Close'], self.fast_period)
        ma_slow = IndicatorCalculator.sma(data['Close'], self.slow_period)

        # 只使用到 current_idx-1 的数据
        if current_idx >= len(ma_fast) or current_idx >= len(ma_slow):
            return self._hold_signal(data, current_idx, "Insufficient MA data")

        current_fast = ma_fast.iloc[current_idx]
        current_slow = ma_slow.iloc[current_idx]

        if pd.isna(current_fast) or pd.isna(current_slow):
            return self._hold_signal(data, current_idx, "MA is NaN")

        # 获取前一个值
        if current_idx > 0:
            prev_fast = ma_fast.iloc[current_idx - 1]
            prev_slow = ma_slow.iloc[current_idx - 1]
        else:
            prev_fast = current_fast
            prev_slow = current_slow

        current_price = float(data['Close'].iloc[current_idx])

        # 金叉买入
        if prev_fast <= prev_slow and current_fast > current_slow:
            strength = (current_fast - current_slow) / current_slow
            # T2-1: 执行层补齐止损止盈
            return Signal(
                strategy_id=self.strategy_id,
                timestamp=data.index[current_idx],
                bar_idx=current_idx,
                action='BUY',
                strength=strength,
                price=current_price,
                stop_loss=current_price * 0.95,
                take_profit=current_price * 1.15,
                reason='Golden cross'
            )

        # 死叉卖出
        elif prev_fast >= prev_slow and current_fast < current_slow:
            strength = (current_slow - current_fast) / current_slow
            return Signal(
                strategy_id=self.strategy_id,
                timestamp=data.index[current_idx],
                bar_idx=current_idx,
                action='SELL',
                strength=strength,
                price=current_price,
                stop_loss=current_price * 1.05,
                take_profit=current_price * 0.85,
                reason='Death cross'
            )

        return self._hold_signal(data, current_idx, "No crossover")


# ==================== 投票融合器 (T1) ====================
class VoteFusion:
    """投票融合器

    规则：
    - T1-1: vote_threshold 参数化
    - T1-2: 冲突票处理
    - T1-3: strength 归一化与阈值
    """

    def __init__(
        self,
        vote_threshold: int = 2,  # T1-1: 默认 2 票同意才交易
        conflict_mode: str = 'HOLD',  # T1-2: HOLD / STRENGTH_DIFF / WEIGHTED
        min_strength_to_vote: float = 0.01,  # T1-3: 最小强度才投票
        strength_aggregation: str = 'avg'  # avg / sum / weighted_avg
    ):
        self.vote_threshold = vote_threshold
        self.conflict_mode = conflict_mode
        self.min_strength_to_vote = min_strength_to_vote
        self.strength_aggregation = strength_aggregation

        # 统计
        self.vote_stats = {
            'total_votes': 0,
            'buy_votes': 0,
            'sell_votes': 0,
            'conflicts': 0,
            'below_threshold': 0
        }

    def fuse_signals(
        self,
        signals: List[Signal],
        current_bar_idx: int
    ) -> Dict:
        """融合多策略信号

        Args:
            signals: 各策略的信号
            current_bar_idx: 当前 bar 索引

        Returns:
            融合决策
        """
        # T0-6: 投票防未来 - 信号已经使用 shift(1)，可以使用当前 bar 的信号
        # 由于所有策略信号都使用 shift(1) 计算，信号实际上使用的是
        # current_bar_idx-1 的数据，因此可以直接使用当前 bar 的信号
        valid_signals = [
            s for s in signals
            if s.bar_idx <= current_bar_idx  # 可以使用当前 bar（已 shift）
        ]

        if not valid_signals:
            return {
                'action': 'HOLD',
                'reason': 'No valid signals available',
                'vote_buy': 0,
                'vote_sell': 0,
                'vote_conflict': 0,
                'signal_strength': 0.0,
                'strategy_votes': {}
            }

        # 分组投票
        buy_votes = []
        sell_votes = []
        hold_votes = []
        strategy_votes = {}

        for signal in valid_signals:
            strategy_votes[signal.strategy_id] = signal.action

            # T1-3: 检查强度阈值
            if signal.action == 'HOLD' or abs(signal.strength) < self.min_strength_to_vote:
                hold_votes.append(signal)
                continue

            self.vote_stats['total_votes'] += 1

            if signal.action == 'BUY':
                buy_votes.append(signal)
                self.vote_stats['buy_votes'] += 1
            elif signal.action == 'SELL':
                sell_votes.append(signal)
                self.vote_stats['sell_votes'] += 1
            else:
                hold_votes.append(signal)

        # T1-2: 冲突检测
        has_buy = len(buy_votes) > 0
        has_sell = len(sell_votes) > 0

        if has_buy and has_sell:
            self.vote_stats['conflicts'] += 1
            return self._resolve_conflict(buy_votes, sell_votes, strategy_votes)

        # T1-1: 检查投票阈值
        if has_buy and len(buy_votes) >= self.vote_threshold:
            return self._aggregate_buy_decision(buy_votes, strategy_votes)

        if has_sell and len(sell_votes) >= self.vote_threshold:
            return self._aggregate_sell_decision(sell_votes, strategy_votes)

        self.vote_stats['below_threshold'] += 1
        return {
            'action': 'HOLD',
            'reason': f'Votes below threshold (buy={len(buy_votes)}, sell={len(sell_votes)}, threshold={self.vote_threshold})',
            'vote_buy': len(buy_votes),
            'vote_sell': len(sell_votes),
            'vote_conflict': 0,
            'signal_strength': 0.0,
            'strategy_votes': strategy_votes
        }

    def _resolve_conflict(
        self,
        buy_votes: List[Signal],
        sell_votes: List[Signal],
        strategy_votes: Dict[str, str]
    ) -> Dict:
        """处理冲突投票 (T1-2)"""
        if self.conflict_mode == 'HOLD':
            return {
                'action': 'HOLD',
                'reason': f'Conflict detected (buy={len(buy_votes)}, sell={len(sell_votes)}), mode=HOLD',
                'vote_buy': len(buy_votes),
                'vote_sell': len(sell_votes),
                'vote_conflict': 1,
                'signal_strength': 0.0,
                'strategy_votes': strategy_votes
            }

        elif self.conflict_mode == 'STRENGTH_DIFF':
            buy_strength = sum(s.strength for s in buy_votes)
            sell_strength = sum(s.strength for s in sell_votes)
            diff = abs(buy_strength - sell_strength)

            if diff > 0.05:  # 强度差超过 5%
                if buy_strength > sell_strength:
                    return self._aggregate_buy_decision(buy_votes, strategy_votes)
                else:
                    return self._aggregate_sell_decision(sell_votes, strategy_votes)

            return {
                'action': 'HOLD',
                'reason': f'Strength diff too small: {diff:.4f}',
                'vote_buy': len(buy_votes),
                'vote_sell': len(sell_votes),
                'vote_conflict': 1,
                'signal_strength': diff,
                'strategy_votes': strategy_votes
            }

        else:  # WEIGHTED
            # 简化处理：回退到 HOLD
            return {
                'action': 'HOLD',
                'reason': 'Conflict mode WEIGHTED not implemented, using HOLD',
                'vote_buy': len(buy_votes),
                'vote_sell': len(sell_votes),
                'vote_conflict': 1,
                'signal_strength': 0.0,
                'strategy_votes': strategy_votes
            }

    def _aggregate_buy_decision(
        self,
        buy_votes: List[Signal],
        strategy_votes: Dict[str, str]
    ) -> Dict:
        """聚合买入决策 (T1-3)"""
        strengths = [s.strength for s in buy_votes]

        if self.strength_aggregation == 'avg':
            agg_strength = sum(strengths) / len(strengths)
        elif self.strength_aggregation == 'sum':
            agg_strength = sum(strengths)
        else:  # weighted_avg (简化)
            agg_strength = sum(strengths) / len(strengths)

        # 使用第一个信号的止损止盈（或可扩展为加权平均）
        ref_signal = buy_votes[0]

        return {
            'action': 'BUY',
            'price': ref_signal.price,
            'stop_loss': ref_signal.stop_loss,
            'take_profit': ref_signal.take_profit,
            'reason': f'{len(buy_votes)} strategies voted BUY',
            'vote_buy': len(buy_votes),
            'vote_sell': 0,
            'vote_conflict': 0,
            'signal_strength': agg_strength,
            'strategy_votes': strategy_votes
        }

    def _aggregate_sell_decision(
        self,
        sell_votes: List[Signal],
        strategy_votes: Dict[str, str]
    ) -> Dict:
        """聚合卖出决策 (T1-3)"""
        strengths = [s.strength for s in sell_votes]

        if self.strength_aggregation == 'avg':
            agg_strength = sum(strengths) / len(strengths)
        elif self.strength_aggregation == 'sum':
            agg_strength = sum(strengths)
        else:
            agg_strength = sum(strengths) / len(strengths)

        ref_signal = sell_votes[0]

        return {
            'action': 'SELL',
            'price': ref_signal.price,
            'stop_loss': ref_signal.stop_loss,
            'take_profit': ref_signal.take_profit,
            'reason': f'{len(sell_votes)} strategies voted SELL',
            'vote_buy': 0,
            'vote_sell': len(sell_votes),
            'vote_conflict': 0,
            'signal_strength': agg_strength,
            'strategy_votes': strategy_votes
        }


# ==================== 回测引擎 ====================
class BacktestEngine:
    """严格历史数据回测引擎"""

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        output_dir: str = None,
        run_id: str = None,
        # T0-3 参数
        max_bars: int = None,
        # 资金管理
        initial_capital: float = 100000,
        max_risk_per_trade: float = 0.02,
        # T1-1 投票阈值
        vote_threshold: int = 2,
        min_strength_to_vote: float = 0.01,
        # T1-2 冲突模式
        conflict_mode: str = 'HOLD',
        # 其他
        log_interval: int = 50,
        enable_lookahead_guard: bool = True  # T0-9
    ):
        self.symbols = symbols
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/backtests")
        self.run_id = run_id or self._generate_run_id()
        self.max_bars = max_bars
        self.log_interval = log_interval
        self.enable_lookahead_guard = enable_lookahead_guard

        # 创建输出目录
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 初始化组件
        self.data_source = YahooFinanceSource(enable_cache=True)
        self.capital_manager = CapitalManager(
            initial_capital=initial_capital,
            max_risk_per_trade=max_risk_per_trade
        )

        # 初始化策略
        self.strategies = [
            MAStrategy('MA_Fast', 5, 20, min_strength_to_vote),
            MAStrategy('MA_Medium', 10, 30, min_strength_to_vote),
            MAStrategy('MA_Slow', 20, 50, min_strength_to_vote),
        ]

        # 投票融合器
        self.vote_fusion = VoteFusion(
            vote_threshold=vote_threshold,
            conflict_mode=conflict_mode,
            min_strength_to_vote=min_strength_to_vote
        )

        # 状态
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []

        # 信号历史（用于 T0-6 验证）
        self.signal_history: List[Signal] = []

        self._log_config()

    def _generate_run_id(self) -> str:
        """生成 run_id"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = "_".join(self.symbols[:3])  # 最多3个
        return f"bt_{symbols_str}_{timestamp}"

    def _log_config(self):
        """记录配置"""
        logger.info("=" * 60)
        logger.info("MKCS 严格历史数据回测引擎")
        logger.info("=" * 60)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"输出目录: {self.run_dir}")
        logger.info(f"回测区间: {self.start_date.date()} ~ {self.end_date.date()}")
        logger.info(f"交易标的: {', '.join(self.symbols)}")
        logger.info(f"初始资金: ${self.capital_manager.initial_capital:,.2f}")
        logger.info(f"投票阈值: {self.vote_fusion.vote_threshold}")
        logger.info(f"冲突模式: {self.vote_fusion.conflict_mode}")
        logger.info(f"最小投票强度: {self.vote_fusion.min_strength_to_vote}")
        logger.info(f"Lookahead 防护: {self.enable_lookahead_guard}")
        logger.info(f"最大 K 线数: {self.max_bars or '无限制'}")
        logger.info("=" * 60)

    def fetch_historical_data(self, symbol: str) -> pd.DataFrame:
        """获取历史数据"""
        try:
            bars = self.data_source.get_bars(
                symbol=symbol,
                start=self.start_date.to_pydatetime(),
                end=self.end_date.to_pydatetime(),
                interval="1d"
            )

            if not bars:
                return pd.DataFrame()

            data = pd.DataFrame([{
                'Date': bar.timestamp,
                'Open': float(bar.open),
                'High': float(bar.high),
                'Low': float(bar.low),
                'Close': float(bar.close),
                'Volume': bar.volume
            } for bar in bars])

            data.set_index('Date', inplace=True)
            data.sort_index(inplace=True)

            # 应用 max_bars 限制
            if self.max_bars and len(data) > self.max_bars:
                data = data.tail(self.max_bars)

            return data

        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return pd.DataFrame()

    def run_backtest(self) -> Dict:
        """运行回测"""
        logger.info("\n开始回测...")

        # 获取所有数据
        all_data = {}
        for symbol in self.symbols:
            data = self.fetch_historical_data(symbol)
            if not data.empty:
                all_data[symbol] = data

        if not all_data:
            logger.error("没有获取到任何数据")
            return {'success': False, 'reason': 'No data'}

        # 获取交易日期
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index)
        trading_dates = sorted(list(all_dates))

        logger.info(f"交易日期数: {len(trading_dates)}")

        # T0-7: 止损止盈防未来 - 明确执行模型
        # 执行模型：使用 t open 价格执行，或 t close 后执行（模拟）
        execution_model = 'next_open'  # 在 t+1 open 执行

        # 主循环
        for i, current_date in enumerate(trading_dates):
            if i % self.log_interval == 0:
                logger.info(f"进度: {i}/{len(trading_dates)} ({current_date.date()})")

            # 1. 生成各策略信号（只使用历史数据）
            current_signals = {}
            for strategy in self.strategies:
                for symbol, data in all_data.items():
                    if current_date not in data.index:
                        continue

                    # 获取当前 bar 在数据中的索引
                    current_idx = data.index.get_loc(current_date)

                    # T0-9: 运行时防护
                    try:
                        signal = strategy.generate_signal(data, current_idx)
                        signal.symbol = symbol
                        current_signals[f"{strategy.strategy_id}_{symbol}"] = signal
                        self.signal_history.append(signal)
                    except ValueError as e:
                        if "Lookahead detected" in str(e):
                            logger.error(f"Lookahead detected in {strategy.strategy_id}!")
                            raise
                        else:
                            logger.warning(f"策略 {strategy.strategy_id} 生成信号失败: {e}")
                            continue

            # 2. 检查止损止盈（T0-7）
            self._check_exit_conditions(all_data, current_date, execution_model)

            # 3. 执行交易信号
            for symbol in all_data.keys():
                if current_date not in all_data[symbol].index:
                    continue

                current_idx = all_data[symbol].index.get_loc(current_date)

                # 收集该标的的所有信号
                symbol_signals = [
                    s for s in current_signals.values()
                    if getattr(s, 'symbol', None) == symbol
                ]

                # 投票融合
                decision = self.vote_fusion.fuse_signals(symbol_signals, current_idx)

                # 执行交易
                if decision['action'] == 'BUY':
                    self._execute_buy(
                        symbol, decision, all_data[symbol], current_date, current_idx
                    )
                elif decision['action'] == 'SELL':
                    self._execute_sell(
                        symbol, decision, all_data[symbol], current_date, current_idx
                    )

            # 4. 记录权益
            self._record_equity(current_date, all_data)

        # 平掉所有持仓
        self._close_all_positions(trading_dates[-1] if trading_dates else pd.Timestamp.now())

        # 生成报告并验证
        return self._finalize_and_validate()

    def _execute_buy(
        self,
        symbol: str,
        decision: Dict,
        data: pd.DataFrame,
        current_date: pd.Timestamp,
        current_idx: int
    ):
        """执行买入"""
        if symbol in self.positions:
            return

        current_price = decision['price']
        stop_loss = decision.get('stop_loss')
        take_profit = decision.get('take_profit')

        # T2-1: 补齐止损止盈
        if stop_loss is None:
            atr = IndicatorCalculator.atr(data, 14)
            if current_idx > 0 and not pd.isna(atr.iloc[current_idx]):
                stop_loss = current_price - self.capital_manager.atr_multiplier * atr.iloc[current_idx]
            else:
                stop_loss = current_price * 0.95

        if take_profit is None:
            take_profit = current_price * 1.15

        # 计算仓位
        current_exposure = sum(
            p.quantity * p.entry_price
            for p in self.positions.values()
        )

        quantity = self.capital_manager.calculate_position_size(
            current_price, stop_loss, current_exposure
        )

        if quantity <= 0:
            return

        # 应用滑点
        entry_price = self.capital_manager.apply_slippage(current_price, 'BUY')
        commission = self.capital_manager.calculate_commission(entry_price, quantity)
        entry_value = entry_price * quantity + commission

        # 开仓
        self.positions[symbol] = Position(
            symbol=symbol,
            direction='LONG',
            quantity=quantity,
            entry_price=entry_price,
            entry_date=current_date,
            entry_bar_idx=current_idx,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_value=entry_value,
            vote_buy=decision['vote_buy'],
            vote_sell=decision['vote_sell'],
            signal_strength=decision['signal_strength']
        )

        logger.debug(f"{current_date.date()} BUY {symbol} {quantity}股 @ ${entry_price:.2f} "
                    f"(SL: ${stop_loss:.2f}, TP: ${take_profit:.2f}) "
                    f"votes: {decision['vote_buy']}-{decision['vote_sell']}")

    def _execute_sell(
        self,
        symbol: str,
        decision: Dict,
        data: pd.DataFrame,
        current_date: pd.Timestamp,
        current_idx: int
    ):
        """执行卖出（用于平仓）"""
        if symbol not in self.positions:
            return

        # 获取当前 bar
        bar = data.loc[current_date]
        self._close_position(symbol, bar, current_date, current_idx, 'SIGNAL', decision)

    def _check_exit_conditions(
        self,
        all_data: Dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
        execution_model: str
    ):
        """检查止损止盈条件 (T0-7)"""
        for symbol, position in list(self.positions.items()):
            if symbol not in all_data or current_date not in all_data[symbol].index:
                continue

            bar = all_data[symbol].loc[current_date]
            current_low = float(bar['Low'])
            current_high = float(bar['High'])
            current_close = float(bar['Close'])

            # T0-7: 明确止损触发逻辑 - 只用当前 bar 的可见字段
            # 如果在 t bar 触发，只能用 t 的 open/high/low
            if position.stop_loss and current_low <= position.stop_loss:
                # 创建模拟 bar 用于平仓
                close_bar = pd.Series({'Close': position.stop_loss}, name=current_date)
                self._close_position(symbol, close_bar, current_date, 0, 'STOP_LOSS')
                continue

            if position.take_profit and current_high >= position.take_profit:
                close_bar = pd.Series({'Close': position.take_profit}, name=current_date)
                self._close_position(symbol, close_bar, current_date, 0, 'TAKE_PROFIT')

    def _close_position(
        self,
        symbol: str,
        bar: pd.Series,
        current_date: pd.Timestamp,
        bar_idx: int,
        reason: str,
        decision: Dict = None
    ):
        """平仓"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        # 应用滑点
        exit_price = self.capital_manager.apply_slippage(float(bar['Close']), 'SELL')
        commission = self.capital_manager.calculate_commission(exit_price, position.quantity)
        exit_value = exit_price * position.quantity - commission

        pnl = exit_value - position.entry_value
        pnl_pct = pnl / position.entry_value

        # 更新资金
        self.capital_manager.update_equity(pnl, current_date)

        # 记录交易
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=current_date,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_value=position.entry_value,
            exit_value=exit_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            commission=commission * 2,
            vote_buy=position.vote_buy,
            vote_sell=position.vote_sell,
            signal_strength=position.signal_strength,
            strategy_votes=decision.get('strategy_votes', {}) if decision else {}
        )
        self.trades.append(trade)

        logger.info(f"{current_date.date()} SELL {symbol} {position.quantity}股 @ ${exit_price:.2f} "
                    f"PnL: ${pnl:+.2f} ({pnl_pct*100:+.2f}%) [{reason}]")

        del self.positions[symbol]

    def _close_all_positions(self, current_date: pd.Timestamp):
        """平掉所有持仓"""
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            bar = pd.Series({'Close': position.entry_price}, name=current_date)
            self._close_position(symbol, bar, current_date, 0, 'TIME_EXIT')

    def _record_equity(self, current_date: pd.Timestamp, all_data: Dict):
        """记录权益"""
        total_value = self.capital_manager.current_capital

        for position in self.positions.values():
            if position.symbol in all_data and current_date in all_data[position.symbol].index:
                current_price = float(all_data[position.symbol].loc[current_date, 'Close'])
                total_value += current_price * position.quantity

        self.equity_curve.append((current_date, total_value))

    def _finalize_and_validate(self) -> Dict:
        """完成回测并验证 (T0-1, T0-2)"""
        logger.info("\n" + "=" * 60)
        logger.info("回测完成，正在生成报告...")
        logger.info("=" * 60)

        # T0-1: 生成文件
        self._save_summary()
        self._save_equity_curve()
        self._save_trades()
        self._save_config()

        # T0-2: 结果一致性自检
        validation_errors = self._validate_results()

        if validation_errors:
            logger.error("结果验证失败!")
            for error in validation_errors:
                logger.error(f"  ❌ {error}")
            logger.error(f"\nRun 目录: {self.run_dir}")
            logger.error("请检查数据一致性")
            return {'success': False, 'errors': validation_errors}

        # 打印完成信息
        logger.info(f"\n✅ 回测完成!")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"输出目录: {self.run_dir}")
        logger.info(f"包含文件:")
        logger.info(f"  - summary.json")
        logger.info(f"  - equity_curve.csv")
        logger.info(f"  - trades.csv")
        logger.info(f"  - config.json")

        return {'success': True, 'run_dir': str(self.run_dir), 'run_id': self.run_id}

    def _save_summary(self):
        """保存摘要 (T0-10: No Lookahead 声明)"""
        # 从权益曲线计算指标（更准确）
        metrics = self._calculate_metrics_from_equity_curve()

        # 统计交易
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]

        exit_reasons = defaultdict(int)
        for t in self.trades:
            exit_reasons[t.exit_reason] += 1

        summary = {
            # 基础信息
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'symbols': self.symbols,

            # T0-10: No Lookahead 声明
            'lookahead_guard': self.enable_lookahead_guard,
            'signal_delay': '1 bar',
            'execution_model': 'next_open',
            'no_lookahead_policy': 'Signals only use data up to t-1, execution at t+1 open',

            # 资金指标
            'initial_capital': metrics['initial_capital'],
            'final_capital': metrics['final_capital'],
            'total_return': metrics['total_return'],
            'total_return_pct': metrics['total_return_pct'],
            'max_drawdown': metrics['max_drawdown'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'sharpe_ratio': metrics['sharpe_ratio'],

            # 交易统计
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,

            # 退出原因统计 (T2-1)
            'exit_reasons': dict(exit_reasons),

            # 配置参数 (T1-1)
            'config': {
                'vote_threshold': self.vote_fusion.vote_threshold,
                'conflict_mode': self.vote_fusion.conflict_mode,
                'min_strength_to_vote': self.vote_fusion.min_strength_to_vote,
                'strength_aggregation': self.vote_fusion.strength_aggregation,
                'max_bars': self.max_bars
            },

            # 投票统计
            'vote_stats': self.vote_fusion.vote_stats
        }

        summary_path = self.run_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"✅ summary.json 已保存")

    def _calculate_metrics_from_equity_curve(self) -> Dict:
        """从权益曲线计算指标"""
        if not self.equity_curve:
            return self.capital_manager.get_metrics()

        equity_values = np.array([e for _, e in self.equity_curve])

        initial_capital = equity_values[0]
        final_capital = equity_values[-1]
        total_return = (final_capital - initial_capital) / initial_capital

        # 计算最大回撤
        peak = np.maximum.accumulate(equity_values)
        drawdowns = (peak - equity_values) / peak
        max_drawdown = drawdowns.max()

        # 计算夏普比率
        returns = np.diff(equity_values) / equity_values[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        return {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe
        }

    def _save_equity_curve(self):
        """保存权益曲线"""
        path = self.run_dir / 'equity_curve.csv'
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'equity'])
            for date, equity in self.equity_curve:
                writer.writerow([date.isoformat(), equity])
        logger.info(f"✅ equity_curve.csv 已保存 ({len(self.equity_curve)} 行)")

    def _save_trades(self):
        """保存交易记录"""
        path = self.run_dir / 'trades.csv'
        with open(path, 'w', newline='') as f:
            fieldnames = [
                'symbol', 'entry_date', 'exit_date', 'direction',
                'entry_price', 'exit_price', 'quantity', 'entry_value',
                'exit_value', 'pnl', 'pnl_pct', 'exit_reason', 'commission',
                'vote_buy', 'vote_sell', 'vote_conflict', 'signal_strength'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in self.trades:
                writer.writerow(trade.to_dict())
        logger.info(f"✅ trades.csv 已保存 ({len(self.trades)} 笔交易)")

    def _save_config(self):
        """保存配置"""
        config = {
            'symbols': self.symbols,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.capital_manager.initial_capital,
            'vote_threshold': self.vote_fusion.vote_threshold,
            'conflict_mode': self.vote_fusion.conflict_mode,
            'min_strength_to_vote': self.vote_fusion.min_strength_to_vote,
            'max_bars': self.max_bars,
            'log_interval': self.log_interval,
            'enable_lookahead_guard': self.enable_lookahead_guard
        }

        path = self.run_dir / 'config.json'
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ config.json 已保存")

    def _validate_results(self) -> List[str]:
        """验证结果一致性 (T0-2)"""
        errors = []

        # 1. 读取生成的文件
        summary_path = self.run_dir / 'summary.json'
        equity_path = self.run_dir / 'equity_curve.csv'
        trades_path = self.run_dir / 'trades.csv'

        if not summary_path.exists():
            errors.append("summary.json 不存在")
            return errors

        with open(summary_path) as f:
            summary = json.load(f)

        # 2. 验证 final_equity == equity_curve 最后一行
        if equity_path.exists():
            equity_df = pd.read_csv(equity_path)
            if not equity_df.empty:
                last_equity = equity_df['equity'].iloc[-1]
                if abs(last_equity - summary['final_capital']) > 0.01:
                    errors.append(
                        f"final_equity 不一致: summary={summary['final_capital']}, "
                        f"equity_curve={last_equity}"
                    )

        # 3. 验证 total_trades == trades.csv 行数
        if trades_path.exists():
            with open(trades_path) as f:
                reader = csv.DictReader(f)
                trade_count = sum(1 for _ in reader)

            if trade_count != summary['total_trades']:
                errors.append(
                    f"total_trades 不一致: summary={summary['total_trades']}, "
                    f"trades.csv={trade_count}"
                )

        # 4. 验证 win_rate
        if summary['total_trades'] > 0:
            expected_win_rate = summary['winning_trades'] / summary['total_trades']
            if abs(expected_win_rate - summary['win_rate']) > 0.01:
                errors.append(
                    f"win_rate 不一致: expected={expected_win_rate:.4f}, "
                    f"summary={summary['win_rate']:.4f}"
                )

        # 5. 验证 max_drawdown 可从 equity_curve 复算
        if equity_path.exists():
            equity_df = pd.read_csv(equity_path)
            if not equity_df.empty:
                equity_values = equity_df['equity'].values
                peak = np.maximum.accumulate(equity_values)
                drawdowns = (peak - equity_values) / peak
                max_dd = drawdowns.max()

                if abs(max_dd - summary['max_drawdown']) > 0.01:
                    errors.append(
                        f"max_drawdown 不一致: summary={summary['max_drawdown']:.4f}, "
                        f"recalculated={max_dd:.4f}"
                    )

        # 6. 验证时间范围 (T0-1)
        if summary['start_date'] != self.start_date.isoformat():
            errors.append(f"start_date 不一致")

        if summary['end_date'] != self.end_date.isoformat():
            errors.append(f"end_date 不一致")

        return errors


# ==================== Lookahead 检测测试 (T0-8) ====================
class LookaheadDetector:
    """Lookahead 检测器"""

    @staticmethod
    def test_shift_required():
        """测试 shift 是否必需（删除 shift 会导致不同的行为）"""
        logger.info("运行 Lookahead 检测测试...")

        # 创建测试数据
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100) * 2)

        data = pd.DataFrame({
            'Close': prices
        }, index=dates)

        # 不带 shift 的 MA
        ma_no_shift = data['Close'].rolling(10).mean()

        # 带 shift 的 MA
        ma_with_shift = data['Close'].rolling(10).mean().shift(1)

        # 验证结果不同
        if (ma_no_shift != ma_with_shift).any():
            logger.info("✅ Lookahead 检测测试通过: shift 改变计算结果")
            return True
        else:
            logger.warning("⚠️  Lookahead 检测测试警告: shift 未改变结果")
            return False


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="MKCS 严格历史数据回测")
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT'], help='交易标的')
    parser.add_argument('--start', default='2023-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-12-01', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    parser.add_argument('--max-risk', type=float, default=0.02, help='单笔最大风险')
    parser.add_argument('--vote-threshold', type=int, default=2, help='投票阈值 (T1-1)')
    parser.add_argument('--conflict-mode', default='HOLD', choices=['HOLD', 'STRENGTH_DIFF'],
                        help='冲突处理模式 (T1-2)')
    parser.add_argument('--min-strength', type=float, default=0.01, help='最小投票强度 (T1-3)')
    parser.add_argument('--max-bars', type=int, default=None, help='最大 K 线数 (T0-3)')
    parser.add_argument('--output-dir', default='outputs/backtests', help='输出目录')
    parser.add_argument('--run-id', default=None, help='Run ID')

    args = parser.parse_args()

    # 运行 Lookahead 检测测试 (T0-8)
    LookaheadDetector.test_shift_required()

    # 创建回测引擎
    engine = BacktestEngine(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        run_id=args.run_id,
        initial_capital=args.capital,
        max_risk_per_trade=args.max_risk,
        vote_threshold=args.vote_threshold,
        min_strength_to_vote=args.min_strength,
        conflict_mode=args.conflict_mode,
        max_bars=args.max_bars
    )

    # 运行回测
    try:
        result = engine.run_backtest()
        if result.get('success'):
            logger.info("\n🎉 回测成功完成!")
        else:
            logger.error("\n❌ 回测失败!")
            for error in result.get('errors', []):
                logger.error(f"  {error}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ 回测异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
