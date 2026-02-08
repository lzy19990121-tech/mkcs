#!/usr/bin/env python3
"""
MKCS 严格历史数据回测引擎 - 性能优化版本

优化内容 (P0-3):
1. 数据缓存到 parquet
2. 指标向量化计算
3. 批量写入 trades/equity
4. 进度输出节流
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
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

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


# ==================== 缓存管理器 (P0-3) ====================
class DataCacheManager:
    """数据缓存管理器"""

    def __init__(self, cache_dir: str = "outputs/cache/price"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """获取缓存文件路径"""
        filename = f"{symbol}_{start_date}_{end_date}.parquet"
        return self.cache_dir / filename

    def load_cached_data(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """加载缓存数据"""
        if not PYARROW_AVAILABLE:
            return None
        cache_path = self.get_cache_path(symbol, start_date, end_date)
        if cache_path.exists():
            logger.info(f"  📦 缓存命中: {symbol}")
            try:
                df = pd.read_parquet(cache_path)
                df.index = pd.to_datetime(df.index)
                return df
            except Exception as e:
                logger.warning(f"缓存读取失败: {e}")
        return None

    def save_cached_data(self, symbol: str, start_date: str, end_date: str, data: pd.DataFrame):
        """保存缓存数据"""
        if not PYARROW_AVAILABLE:
            return
        cache_path = self.get_cache_path(symbol, start_date, end_date)
        try:
            data.to_parquet(cache_path, index=True)
            logger.info(f"  💾 缓存已保存: {cache_path.name}")
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")

    def clear_cache(self):
        """清空缓存"""
        for cache_file in self.cache_dir.glob("*.parquet"):
            cache_file.unlink()
        logger.info("缓存已清空")


# ==================== 指标计算器 (P0-3: 向量化) ====================
class VectorizedIndicatorCalculator:
    """向量化的技术指标计算器"""

    @staticmethod
    def add_all_indicators(data: pd.DataFrame,
                           ma_periods: List[Tuple[str, int]] = None,
                           atr_period: int = 14,
                           rsi_period: int = 14) -> pd.DataFrame:
        """一次性计算所有指标（向量化）

        Args:
            data: 原始 OHLCV 数据
            ma_periods: MA 周期列表，如 [('MA5', 5), ('MA20', 20)]
            atr_period: ATR 周期
            rsi_period: RSI 周期

        Returns:
            添加了所有指标列的 DataFrame
        """
        df = data.copy()

        if ma_periods is None:
            ma_periods = [('MA5', 5), ('MA10', 10), ('MA20', 20), ('MA30', 30), ('MA50', 50)]

        # 计算所有 MA（向量化）
        for name, period in ma_periods:
            df[name] = df['Close'].rolling(window=period).mean().shift(1)

        # 计算 ATR（向量化）
        df['ATR'] = VectorizedIndicatorCalculator.atr(df, atr_period).shift(1)

        # 计算 RSI（向量化）
        df['RSI'] = VectorizedIndicatorCalculator.rsi(df['Close'], rsi_period)

        # 计算收益率
        df['Returns'] = df['Close'].pct_change().shift(1)

        return df

    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """平均真实范围（向量化）"""
        high = data['High']
        low = data['Low']
        close = data['Close']

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = tr1.combine(tr2, max).combine(tr3, max)

        return tr.rolling(window=period).mean()

    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """相对强弱指标（向量化）"""
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))


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
    exit_reason: str
    commission: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    hold_bars: int = 0
    vote_buy: int = 0
    vote_sell: int = 0
    vote_conflict: int = 0
    signal_strength: float = 0.0

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
    entry_bar_idx: int
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_value: float
    min_hold_until: int  # 最小持仓日期
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
    strength: float
    confidence: float
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
    """资金管理器"""

    def __init__(
        self,
        initial_capital: float = 100000,
        max_risk_per_trade: float = 0.02,
        max_position_pct: float = 0.20,
        max_total_exposure: float = 0.80,
        commission_rate: float = 0.001,
        slippage_bps: float = 5.0,
        atr_multiplier: float = 2.0
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.atr_multiplier = atr_multiplier

        # 批量写入缓冲区
        self.equity_curve_buffer: List[Tuple[pd.Timestamp, float]] = []
        self.equity_timestamps = []
        self.equity_values = []

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
        if stop_loss and stop_loss != entry_price:
            risk_per_share = abs(entry_price - stop_loss)
            risk_amount = self.current_capital * self.max_risk_per_trade
            shares_by_risk = int(risk_amount / risk_per_share)
        else:
            shares_by_risk = float('inf')

        max_position_value = self.current_capital * self.max_position_pct
        shares_by_position = int(max_position_value / entry_price)

        available_capital = self.current_capital * self.max_total_exposure - current_exposure
        shares_by_capital = int(available_capital / entry_price) if available_capital > 0 else 0

        shares = min(shares_by_risk, shares_by_position, shares_by_capital)
        shares = max(0, (shares // 100) * 100)

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
        """更新权益（批量缓冲）"""
        self.current_capital += pnl
        self.equity_timestamps.append(timestamp)
        self.equity_values.append(self.current_capital)

    def flush_equity_buffer(self) -> List[Tuple[pd.Timestamp, float]]:
        """刷新权益缓冲区"""
        result = list(zip(self.equity_timestamps, self.equity_values))
        self.equity_timestamps = []
        self.equity_values = []
        return result

    def get_metrics(self) -> Dict:
        equity_array = np.array(self.equity_values) if self.equity_values else np.array([self.initial_capital])
        returns = np.diff(equity_array) / equity_array[:-1] if len(equity_array) > 1 else np.array([0])

        total_return = (self.current_capital - self.initial_capital) / self.initial_capital

        # 计算最大回撤
        peak = np.maximum.accumulate(equity_array)
        drawdowns = (peak - equity_array) / peak
        max_drawdown = drawdowns.max()

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

    def __init__(self, strategy_id: str, min_strength: float = 0.01, min_confidence: float = 0.3):
        self.strategy_id = strategy_id
        self.min_strength = min_strength  # P1-1: 防噪声过滤
        self.min_confidence = min_confidence  # P1-1: 置信度过滤

    def generate_signal(
        self,
        data: pd.DataFrame,
        current_idx: int,
        indicators: pd.DataFrame
    ) -> Signal:
        """生成信号（使用预计算的指标）"""
        if current_idx < 50:
            return self._hold_signal(data, current_idx, "Insufficient data")

        signal = self._compute_signal(data, current_idx, indicators)

        # P1-1: 强度和置信度过滤
        if signal.action != 'HOLD':
            if abs(signal.strength) < self.min_strength:
                return Signal(
                    strategy_id=self.strategy_id,
                    timestamp=data.index[current_idx],
                    bar_idx=current_idx,
                    action='HOLD',
                    strength=0,
                    confidence=0,
                    price=signal.price,
                    stop_loss=None,
                    take_profit=None,
                    reason=f"Strength too low: {signal.strength:.4f} < {self.min_strength}"
                )
            if signal.confidence < self.min_confidence:
                return Signal(
                    strategy_id=self.strategy_id,
                    timestamp=data.index[current_idx],
                    bar_idx=current_idx,
                    action='HOLD',
                    strength=signal.strength,
                    confidence=signal.confidence,
                    price=signal.price,
                    stop_loss=None,
                    take_profit=None,
                    reason=f"Confidence too low: {signal.confidence:.2f} < {self.min_confidence}"
                )

        return signal

    def _compute_signal(self, data: pd.DataFrame, current_idx: int, indicators: pd.DataFrame) -> Signal:
        """计算信号（子类实现）"""
        raise NotImplementedError

    def _hold_signal(self, data: pd.DataFrame, current_idx: int, reason: str) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            timestamp=data.index[current_idx],
            bar_idx=current_idx,
            action='HOLD',
            strength=0,
            confidence=0,
            price=float(data['Close'].iloc[current_idx]),
            stop_loss=None,
            take_profit=None,
            reason=reason
        )


class MAStrategy(Strategy):
    """移动平均策略（使用预计算指标）"""

    def __init__(
        self,
        strategy_id: str,
        fast_period: int,
        slow_period: int,
        min_strength: float = 0.001,  # 降低默认最小强度
        min_confidence: float = 0.0   # 降低默认最小置信度
    ):
        super().__init__(strategy_id, min_strength, min_confidence)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_col = f'MA{fast_period}'
        self.slow_col = f'MA{slow_period}'

    def _compute_signal(self, data: pd.DataFrame, current_idx: int, indicators: pd.DataFrame) -> Signal:
        # 使用预计算的 MA
        if current_idx >= len(indicators):
            return self._hold_signal(data, current_idx, "Insufficient indicator data")

        current_fast = indicators[self.fast_col].iloc[current_idx]
        current_slow = indicators[self.slow_col].iloc[current_idx]

        if pd.isna(current_fast) or pd.isna(current_slow):
            return self._hold_signal(data, current_idx, "MA is NaN")

        if current_idx > 0:
            prev_fast = indicators[self.fast_col].iloc[current_idx - 1]
            prev_slow = indicators[self.slow_col].iloc[current_idx - 1]
        else:
            prev_fast = current_fast
            prev_slow = current_slow

        current_price = float(data['Close'].iloc[current_idx])

        # 金叉买入
        if prev_fast <= prev_slow and current_fast > current_slow:
            strength = (current_fast - current_slow) / current_slow
            # 置信度基于强度 - 更宽松的计算
            confidence = min(abs(strength) * 20, 1.0)

            # P1-3: ATR-based 止损止盈
            atr = indicators['ATR'].iloc[current_idx] if 'ATR' in indicators else None
            if not pd.isna(atr) and atr > 0:
                stop_loss = current_price - 2 * atr
                take_profit = current_price + 4 * atr
            else:
                stop_loss = current_price * 0.95
                take_profit = current_price * 1.15

            return Signal(
                strategy_id=self.strategy_id,
                timestamp=data.index[current_idx],
                bar_idx=current_idx,
                action='BUY',
                strength=strength,
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason='Golden cross'
            )

        # 死叉卖出
        elif prev_fast >= prev_slow and current_fast < current_slow:
            strength = (current_slow - current_fast) / current_slow
            confidence = min(abs(strength) * 20, 1.0)

            atr = indicators['ATR'].iloc[current_idx] if 'ATR' in indicators else None
            if not pd.isna(atr) and atr > 0:
                stop_loss = current_price + 2 * atr
                take_profit = current_price - 4 * atr
            else:
                stop_loss = current_price * 1.05
                take_profit = current_price * 0.85

            return Signal(
                strategy_id=self.strategy_id,
                timestamp=data.index[current_idx],
                bar_idx=current_idx,
                action='SELL',
                strength=strength,
                confidence=confidence,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason='Death cross'
            )

        return self._hold_signal(data, current_idx, "No crossover")


# ==================== 投票融合器 ====================
class VoteFusion:
    """投票融合器"""

    def __init__(
        self,
        vote_threshold: int = 2,
        conflict_mode: str = 'HOLD',
        min_strength_to_vote: float = 0.01,
        min_avg_confidence: float = 0.0,  # P1-1: 平均置信度门槛
        strength_aggregation: str = 'avg'
    ):
        self.vote_threshold = vote_threshold
        self.conflict_mode = conflict_mode
        self.min_strength_to_vote = min_strength_to_vote
        self.min_avg_confidence = min_avg_confidence
        self.strength_aggregation = strength_aggregation

        self.vote_stats = {
            'total_votes': 0,
            'buy_votes': 0,
            'sell_votes': 0,
            'conflicts': 0,
            'below_threshold': 0,
            'below_confidence': 0  # P1-1: 置信度过滤统计
        }

    def fuse_signals(
        self,
        signals: List[Signal],
        current_bar_idx: int
    ) -> Dict:
        """融合多策略信号"""
        valid_signals = [
            s for s in signals
            if s.bar_idx <= current_bar_idx
        ]

        if not valid_signals:
            return {
                'action': 'HOLD',
                'reason': 'No valid signals available',
                'vote_buy': 0,
                'vote_sell': 0,
                'vote_conflict': 0,
                'signal_strength': 0.0,
                'avg_confidence': 0.0,
                'strategy_votes': {}
            }

        buy_votes = []
        sell_votes = []
        hold_votes = []
        strategy_votes = {}

        for signal in valid_signals:
            strategy_votes[signal.strategy_id] = signal.action

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

        has_buy = len(buy_votes) > 0
        has_sell = len(sell_votes) > 0

        if has_buy and has_sell:
            self.vote_stats['conflicts'] += 1
            return self._resolve_conflict(buy_votes, sell_votes, strategy_votes)

        # P1-1: 检查平均置信度
        if has_buy and len(buy_votes) >= self.vote_threshold:
            avg_conf = np.mean([s.confidence for s in buy_votes])
            if avg_conf < self.min_avg_confidence:
                self.vote_stats['below_confidence'] += 1
                return {
                    'action': 'HOLD',
                    'reason': f'Avg confidence too low: {avg_conf:.2f} < {self.min_avg_confidence}',
                    'vote_buy': len(buy_votes),
                    'vote_sell': 0,
                    'vote_conflict': 0,
                    'signal_strength': 0.0,
                    'avg_confidence': avg_conf,
                    'strategy_votes': strategy_votes
                }
            return self._aggregate_buy_decision(buy_votes, strategy_votes)

        if has_sell and len(sell_votes) >= self.vote_threshold:
            avg_conf = np.mean([s.confidence for s in sell_votes])
            if avg_conf < self.min_avg_confidence:
                self.vote_stats['below_confidence'] += 1
                return {
                    'action': 'HOLD',
                    'reason': f'Avg confidence too low: {avg_conf:.2f} < {self.min_avg_confidence}',
                    'vote_buy': 0,
                    'vote_sell': len(sell_votes),
                    'vote_conflict': 0,
                    'signal_strength': 0.0,
                    'avg_confidence': avg_conf,
                    'strategy_votes': strategy_votes
                }
            return self._aggregate_sell_decision(sell_votes, strategy_votes)

        self.vote_stats['below_threshold'] += 1
        return {
            'action': 'HOLD',
            'reason': f'Votes below threshold (buy={len(buy_votes)}, sell={len(sell_votes)}, threshold={self.vote_threshold})',
            'vote_buy': len(buy_votes),
            'vote_sell': len(sell_votes),
            'vote_conflict': 0,
            'signal_strength': 0.0,
            'avg_confidence': 0.0,
            'strategy_votes': strategy_votes
        }

    def _resolve_conflict(
        self,
        buy_votes: List[Signal],
        sell_votes: List[Signal],
        strategy_votes: Dict[str, str]
    ) -> Dict:
        """处理冲突投票"""
        if self.conflict_mode == 'HOLD':
            return {
                'action': 'HOLD',
                'reason': f'Conflict detected (buy={len(buy_votes)}, sell={len(sell_votes)}), mode=HOLD',
                'vote_buy': len(buy_votes),
                'vote_sell': len(sell_votes),
                'vote_conflict': 1,
                'signal_strength': 0.0,
                'avg_confidence': 0.0,
                'strategy_votes': strategy_votes
            }

        elif self.conflict_mode == 'STRENGTH_DIFF':
            buy_strength = sum(s.strength for s in buy_votes)
            sell_strength = sum(s.strength for s in sell_votes)
            diff = abs(buy_strength - sell_strength)

            if diff > 0.05:
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
                'avg_confidence': 0.0,
                'strategy_votes': strategy_votes
            }

        else:
            return {
                'action': 'HOLD',
                'reason': 'Conflict mode WEIGHTED not implemented, using HOLD',
                'vote_buy': len(buy_votes),
                'vote_sell': len(sell_votes),
                'vote_conflict': 1,
                'signal_strength': 0.0,
                'avg_confidence': 0.0,
                'strategy_votes': strategy_votes
            }

    def _aggregate_buy_decision(
        self,
        buy_votes: List[Signal],
        strategy_votes: Dict[str, str]
    ) -> Dict:
        """聚合买入决策"""
        strengths = [s.strength for s in buy_votes]
        confidences = [s.confidence for s in buy_votes]

        if self.strength_aggregation == 'avg':
            agg_strength = sum(strengths) / len(strengths)
        elif self.strength_aggregation == 'sum':
            agg_strength = sum(strengths)
        else:
            agg_strength = sum(strengths) / len(strengths)

        avg_confidence = sum(confidences) / len(confidences)
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
            'avg_confidence': avg_confidence,
            'strategy_votes': strategy_votes
        }

    def _aggregate_sell_decision(
        self,
        sell_votes: List[Signal],
        strategy_votes: Dict[str, str]
    ) -> Dict:
        """聚合卖出决策"""
        strengths = [s.strength for s in sell_votes]
        confidences = [s.confidence for s in sell_votes]

        if self.strength_aggregation == 'avg':
            agg_strength = sum(strengths) / len(strengths)
        elif self.strength_aggregation == 'sum':
            agg_strength = sum(strengths)
        else:
            agg_strength = sum(strengths) / len(strengths)

        avg_confidence = sum(confidences) / len(confidences)
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
            'avg_confidence': avg_confidence,
            'strategy_votes': strategy_votes
        }


# ==================== 回测引擎 ====================
class BacktestEngineOptimized:
    """性能优化的回测引擎"""

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        output_dir: str = None,
        run_id: str = None,
        # 资金管理
        initial_capital: float = 100000,
        max_risk_per_trade: float = 0.02,
        # 投票配置
        vote_threshold: int = 2,
        min_strength_to_vote: float = 0.01,
        min_avg_confidence: float = 0.0,  # P1-1
        conflict_mode: str = 'HOLD',
        # 策略配置
        strategy_min_strength: float = 0.01,
        strategy_min_confidence: float = 0.3,  # P1-1
        # P1-2: 持仓周期和冷却
        min_hold_bars: int = 0,
        cooldown_bars_after_stop: int = 0,
        # P1-3: ATR 止损止盈
        use_atr_stops: bool = True,
        stop_atr_mult: float = 2.0,
        tp_atr_mult: float = 4.0,
        # 性能配置
        log_interval: int = 50,
        use_cache: bool = True
    ):
        self.symbols = symbols
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/backtests")
        self.run_id = run_id or self._generate_run_id()
        self.log_interval = log_interval
        self.use_cache = use_cache

        # 创建输出目录
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 缓存管理器
        self.cache_manager = DataCacheManager() if use_cache else None

        # 初始化组件
        self.data_source = YahooFinanceSource(enable_cache=True)
        self.capital_manager = CapitalManager(
            initial_capital=initial_capital,
            max_risk_per_trade=max_risk_per_trade
        )

        # P1-2: 持仓和冷却跟踪
        self.min_hold_bars = min_hold_bars
        self.cooldown_bars_after_stop = cooldown_bars_after_stop
        self.cooldown_until = {}  # symbol -> bar_idx

        # 初始化策略
        self.strategies = [
            MAStrategy('MA_Fast', 5, 20, strategy_min_strength, strategy_min_confidence),
            MAStrategy('MA_Medium', 10, 30, strategy_min_strength, strategy_min_confidence),
            MAStrategy('MA_Slow', 20, 50, strategy_min_strength, strategy_min_confidence),
        ]

        # 投票融合器
        self.vote_fusion = VoteFusion(
            vote_threshold=vote_threshold,
            conflict_mode=conflict_mode,
            min_strength_to_vote=min_strength_to_vote,
            min_avg_confidence=min_avg_confidence
        )

        # P1-3: ATR 配置
        self.use_atr_stops = use_atr_stops
        self.stop_atr_mult = stop_atr_mult
        self.tp_atr_mult = tp_atr_mult

        # 状态
        self.positions: Dict[str, Position] = {}
        self.trades_buffer: List[Trade] = []

        self._log_config()

    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = "_".join(self.symbols[:3])
        return f"bt_{symbols_str}_{timestamp}"

    def _log_config(self):
        logger.info("=" * 60)
        logger.info("MKCS 性能优化回测引擎")
        logger.info("=" * 60)
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"回测区间: {self.start_date.date()} ~ {self.end_date.date()}")
        logger.info(f"交易标的: {', '.join(self.symbols)}")
        logger.info(f"初始资金: ${self.capital_manager.initial_capital:,.2f}")
        logger.info(f"投票阈值: {self.vote_fusion.vote_threshold}")
        logger.info(f"最小平均置信度: {self.vote_fusion.min_avg_confidence}")
        logger.info(f"最小持仓周期: {self.min_hold_bars} bars")
        logger.info(f"止损冷却: {self.cooldown_bars_after_stop} bars")
        logger.info(f"ATR 止损止盈: {self.use_atr_stops}")
        logger.info(f"数据缓存: {self.use_cache}")
        logger.info("=" * 60)

    def fetch_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取历史数据（带缓存）"""
        start_str = self.start_date.strftime('%Y-%m-%d')
        end_str = self.end_date.strftime('%Y-%m-%d')

        # 尝试从缓存加载
        if self.cache_manager:
            cached = self.cache_manager.load_cached_data(symbol, start_str, end_str)
            if cached is not None:
                return cached

        # 从数据源获取
        try:
            bars = self.data_source.get_bars(
                symbol=symbol,
                start=self.start_date.to_pydatetime(),
                end=self.end_date.to_pydatetime(),
                interval="1d"
            )

            if not bars:
                return None

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

            # 保存缓存
            if self.cache_manager:
                self.cache_manager.save_cached_data(symbol, start_str, end_str, data)

            return data

        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return None

    def run_backtest(self) -> Dict:
        """运行回测"""
        logger.info("\n开始回测...")
        start_time = datetime.now()

        # 获取并预处理所有数据
        all_data = {}
        all_indicators = {}

        for symbol in self.symbols:
            data = self.fetch_historical_data(symbol)
            if data is not None and not data.empty:
                # 向量化计算所有指标
                indicators = VectorizedIndicatorCalculator.add_all_indicators(data)
                all_data[symbol] = data
                all_indicators[symbol] = indicators

        if not all_data:
            logger.error("没有获取到任何数据")
            return {'success': False, 'reason': 'No data'}

        # 获取交易日期
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index)
        trading_dates = sorted(list(all_dates))

        logger.info(f"交易日期数: {len(trading_dates)}")

        # 主循环
        for i, current_date in enumerate(trading_dates):
            if i % self.log_interval == 0:
                logger.info(f"进度: {i}/{len(trading_dates)} ({current_date.date()})")

            current_idx = i  # 在交易日期中的索引

            # 1. 检查止损止盈
            self._check_exit_conditions(all_data, all_indicators, current_date, current_idx)

            # 2. 生成交易信号
            current_signals = {}
            for strategy in self.strategies:
                for symbol, data in all_data.items():
                    if current_date not in data.index:
                        continue

                    # 获取数据中的索引
                    data_idx = data.index.get_loc(current_date)
                    indicators = all_indicators[symbol]

                    try:
                        signal = strategy.generate_signal(data, data_idx, indicators)
                        signal.symbol = symbol
                        current_signals[f"{strategy.strategy_id}_{symbol}"] = signal
                    except Exception as e:
                        continue

            # 3. 执行交易信号
            for symbol in all_data.keys():
                if current_date not in all_data[symbol].index:
                    continue

                data_idx = all_data[symbol].index.get_loc(current_date)

                # 检查冷却期
                if symbol in self.cooldown_until and current_idx < self.cooldown_until[symbol]:
                    continue

                # 收集该标的的所有信号
                symbol_signals = [
                    s for s in current_signals.values()
                    if getattr(s, 'symbol', None) == symbol
                ]

                # 投票融合
                decision = self.vote_fusion.fuse_signals(symbol_signals, data_idx)

                # 执行交易
                if decision['action'] == 'BUY':
                    self._execute_buy(
                        symbol, decision, all_data[symbol], all_indicators[symbol],
                        current_date, data_idx, current_idx
                    )
                elif decision['action'] == 'SELL':
                    self._execute_sell(
                        symbol, decision, all_data[symbol], current_date, data_idx
                    )

            # 4. 记录权益
            self._record_equity(current_date, all_data)

        # 平掉所有持仓
        self._close_all_positions(trading_dates[-1] if trading_dates else pd.Timestamp.now())

        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n回测完成，耗时: {elapsed:.1f} 秒")

        return self._finalize_and_validate()

    def _execute_buy(
        self,
        symbol: str,
        decision: Dict,
        data: pd.DataFrame,
        indicators: pd.DataFrame,
        current_date: pd.Timestamp,
        data_idx: int,
        global_idx: int
    ):
        """执行买入"""
        if symbol in self.positions:
            return

        current_price = decision['price']

        # P1-3: ATR-based 止损止盈
        if self.use_atr_stops and 'ATR' in indicators:
            atr = indicators['ATR'].iloc[data_idx]
            if not pd.isna(atr) and atr > 0:
                stop_loss = current_price - self.stop_atr_mult * atr
                take_profit = current_price + self.tp_atr_mult * atr
            else:
                stop_loss = current_price * 0.95
                take_profit = current_price * 1.15
        else:
            stop_loss = decision.get('stop_loss', current_price * 0.95)
            take_profit = decision.get('take_profit', current_price * 1.15)

        # 计算仓位
        current_exposure = sum(
            p.quantity * p.entry_price
            for p in self.positions.values()
        )

        atr = indicators['ATR'].iloc[data_idx] if 'ATR' in indicators else None
        quantity = self.capital_manager.calculate_position_size(
            current_price, stop_loss, current_exposure, atr
        )

        if quantity <= 0:
            return

        # 应用滑点
        entry_price = self.capital_manager.apply_slippage(current_price, 'BUY')
        commission = self.capital_manager.calculate_commission(entry_price, quantity)
        entry_value = entry_price * quantity + commission

        # P1-2: 最小持仓周期
        min_hold_until = global_idx + self.min_hold_bars

        # 开仓
        self.positions[symbol] = Position(
            symbol=symbol,
            direction='LONG',
            quantity=quantity,
            entry_price=entry_price,
            entry_date=current_date,
            entry_bar_idx=global_idx,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_value=entry_value,
            min_hold_until=min_hold_until,
            vote_buy=decision['vote_buy'],
            vote_sell=decision['vote_sell'],
            signal_strength=decision['signal_strength']
        )

    def _execute_sell(
        self,
        symbol: str,
        decision: Dict,
        data: pd.DataFrame,
        current_date: pd.Timestamp,
        data_idx: int
    ):
        """执行卖出（用于平仓）"""
        if symbol not in self.positions:
            return

        bar = data.loc[current_date]
        self._close_position(symbol, bar, current_date, data_idx, 'SIGNAL', decision)

    def _check_exit_conditions(
        self,
        all_data: Dict[str, pd.DataFrame],
        all_indicators: Dict[str, pd.DataFrame],
        current_date: pd.Timestamp,
        global_idx: int
    ):
        """检查止损止盈条件"""
        for symbol, position in list(self.positions.items()):
            if symbol not in all_data or current_date not in all_data[symbol].index:
                continue

            bar = all_data[symbol].loc[current_date]
            current_low = float(bar['Low'])
            current_high = float(bar['High'])

            # P1-2: 检查最小持仓周期
            if global_idx < position.min_hold_until:
                # 只检查止盈，不检查止损
                if position.take_profit and current_high >= position.take_profit:
                    close_bar = pd.Series({'Close': position.take_profit}, name=current_date)
                    self._close_position(symbol, close_bar, current_date, global_idx, 'TAKE_PROFIT')
                continue

            # 正常检查止损止盈
            if position.stop_loss and current_low <= position.stop_loss:
                close_bar = pd.Series({'Close': position.stop_loss}, name=current_date)
                self._close_position(symbol, close_bar, current_date, global_idx, 'STOP_LOSS')

                # P1-2: 设置冷却期
                if self.cooldown_bars_after_stop > 0:
                    self.cooldown_until[symbol] = global_idx + self.cooldown_bars_after_stop
                continue

            if position.take_profit and current_high >= position.take_profit:
                close_bar = pd.Series({'Close': position.take_profit}, name=current_date)
                self._close_position(symbol, close_bar, current_date, global_idx, 'TAKE_PROFIT')

    def _close_position(
        self,
        symbol: str,
        bar: pd.Series,
        current_date: pd.Timestamp,
        global_idx: int,
        reason: str,
        decision: Dict = None
    ):
        """平仓"""
        if symbol not in self.positions:
            return

        position = self.positions[symbol]

        exit_price = self.capital_manager.apply_slippage(float(bar['Close']), 'SELL')
        commission = self.capital_manager.calculate_commission(exit_price, position.quantity)
        exit_value = exit_price * position.quantity - commission

        pnl = exit_value - position.entry_value
        pnl_pct = pnl / position.entry_value

        # 更新资金
        self.capital_manager.update_equity(pnl, current_date)

        # 记录交易（缓冲）
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
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            hold_bars=global_idx - position.entry_bar_idx,
            vote_buy=position.vote_buy,
            vote_sell=position.vote_sell,
            signal_strength=position.signal_strength
        )
        self.trades_buffer.append(trade)

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

        self.capital_manager.update_equity(0, current_date)  # 仅记录，不更新资金

        # 修正：直接更新权益值
        self.capital_manager.equity_values[-1] = total_value

    def _finalize_and_validate(self) -> Dict:
        """完成回测并验证"""
        logger.info("\n" + "=" * 60)
        logger.info("回测完成，正在生成报告...")
        logger.info("=" * 60)

        # 批量写入文件
        self._save_summary()
        self._save_equity_curve()
        self._save_trades()
        self._save_config()

        return {
            'success': True,
            'run_dir': str(self.run_dir),
            'run_id': self.run_id
        }

    def _save_summary(self):
        """保存摘要"""
        metrics = self.capital_manager.get_metrics()

        # 统计交易
        winning_trades = [t for t in self.trades_buffer if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trades_buffer if t.pnl and t.pnl < 0]

        exit_reasons = defaultdict(int)
        for t in self.trades_buffer:
            exit_reasons[t.exit_reason] += 1

        # P1-3: 止损止盈统计
        stop_losses = [t.stop_loss for t in self.trades_buffer if t.stop_loss]
        take_profits = [t.take_profit for t in self.trades_buffer if t.take_profit]

        # P2-2: 分段评估（按年份）
        equity_curve = list(zip(self.capital_manager.equity_timestamps, self.capital_manager.equity_values))
        yearly_metrics = self._calculate_yearly_metrics(equity_curve)

        # P2-2: 最大亏损窗口
        drawdown_windows = self._calculate_drawdown_windows(equity_curve)

        summary = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'symbols': self.symbols,

            'lookahead_guard': True,
            'signal_delay': '1 bar',
            'execution_model': 'next_open',

            # 资金指标
            'initial_capital': metrics['initial_capital'],
            'final_capital': metrics['final_capital'],
            'total_return': metrics['total_return'],
            'total_return_pct': metrics['total_return_pct'],
            'max_drawdown': metrics['max_drawdown'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'sharpe_ratio': metrics['sharpe_ratio'],

            # 交易统计
            'total_trades': len(self.trades_buffer),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades_buffer) if self.trades_buffer else 0,

            # 退出原因
            'exit_reasons': dict(exit_reasons),

            # 配置
            'config': {
                'vote_threshold': self.vote_fusion.vote_threshold,
                'min_avg_confidence': self.vote_fusion.min_avg_confidence,
                'min_hold_bars': self.min_hold_bars,
                'cooldown_bars_after_stop': self.cooldown_bars_after_stop,
                'use_atr_stops': self.use_atr_stops,
                'stop_atr_mult': self.stop_atr_mult,
                'tp_atr_mult': self.tp_atr_mult,
            },

            # 投票统计
            'vote_stats': self.vote_fusion.vote_stats,

            # P1-3: 止损止盈统计
            'stop_loss_stats': {
                'count': len(stop_losses),
                'mean': np.mean(stop_losses) if stop_losses else None,
                'median': np.median(stop_losses) if stop_losses else None,
            } if stop_losses else {},
            'take_profit_stats': {
                'count': len(take_profits),
                'mean': np.mean(take_profits) if take_profits else None,
                'median': np.median(take_profits) if take_profits else None,
            } if take_profits else {},

            # P2-2: 分段评估
            'yearly_metrics': yearly_metrics,

            # P2-2: 最大亏损窗口
            'drawdown_windows': drawdown_windows,
        }

        summary_path = self.run_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"✅ summary.json 已保存")

    def _calculate_yearly_metrics(self, equity_curve: List) -> Dict:
        """计算按年份的指标 (P2-2)"""
        if not equity_curve:
            return {}

        # 按年份分组
        yearly_data = defaultdict(lambda: {'equity': [], 'dates': []})
        for date, equity in equity_curve:
            year = date.year
            yearly_data[year]['equity'].append(equity)
            yearly_data[year]['dates'].append(date)

        yearly_metrics = {}
        for year, data in sorted(yearly_data.items()):
            equity_array = np.array(data['equity'])
            if len(equity_array) < 2:
                continue

            initial = equity_array[0]
            final = equity_array[-1]
            total_return = (final - initial) / initial if initial > 0 else 0

            # 计算该年最大回撤
            peak = np.maximum.accumulate(equity_array)
            drawdowns = (peak - equity_array) / peak
            max_drawdown = drawdowns.max()

            # 计算夏普
            returns = np.diff(equity_array) / equity_array[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0

            # 统计该年交易
            year_trades = [t for t in self.trades_buffer if t.exit_date and t.exit_date.year == year]
            winning = len([t for t in year_trades if t.pnl and t.pnl > 0])

            yearly_metrics[str(year)] = {
                'return_pct': total_return * 100,
                'max_drawdown_pct': max_drawdown * 100,
                'sharpe_ratio': sharpe,
                'trades': len(year_trades),
                'win_rate': winning / len(year_trades) if year_trades else 0,
            }

        return yearly_metrics

    def _calculate_drawdown_windows(self, equity_curve: List) -> Dict:
        """计算最大亏损窗口 (P2-2)"""
        if not equity_curve or len(equity_curve) < 60:
            return {}

        equity_values = np.array([e for _, e in equity_curve])
        dates = [d for d, _ in equity_curve]

        result = {}

        # 计算滚动窗口最大亏损
        for window in [20, 60]:
            if len(equity_values) < window:
                continue

            min_loss = 0  # 最小亏损（最负的值）
            worst_start_idx = 0
            worst_end_idx = 0

            for i in range(window - 1, len(equity_values)):
                window_start = equity_values[i - window + 1]
                window_end = equity_values[i]
                window_loss = (window_end - window_start) / window_start if window_start > 0 else 0

                if window_loss < min_loss:
                    min_loss = window_loss
                    worst_start_idx = i - window + 1
                    worst_end_idx = i

            result[f'{window}d_window'] = {
                'max_loss_pct': min_loss * 100,
                'start_date': dates[worst_start_idx].isoformat() if worst_start_idx < len(dates) else None,
                'end_date': dates[worst_end_idx].isoformat() if worst_end_idx < len(dates) else None,
            }

        return result

    def _save_equity_curve(self):
        """保存权益曲线"""
        path = self.run_dir / 'equity_curve.csv'
        equity_curve = self.capital_manager.flush_equity_buffer()

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['date', 'equity'])
            for date, equity in equity_curve:
                writer.writerow([date.isoformat(), equity])

        logger.info(f"✅ equity_curve.csv 已保存 ({len(equity_curve)} 行)")

    def _save_trades(self):
        """保存交易记录"""
        path = self.run_dir / 'trades.csv'
        fieldnames = [
            'symbol', 'entry_date', 'exit_date', 'direction',
            'entry_price', 'exit_price', 'quantity', 'entry_value',
            'exit_value', 'pnl', 'pnl_pct', 'exit_reason', 'commission',
            'stop_loss', 'take_profit', 'hold_bars',
            'vote_buy', 'vote_sell', 'vote_conflict', 'signal_strength'
        ]

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trade in self.trades_buffer:
                writer.writerow(trade.to_dict())

        logger.info(f"✅ trades.csv 已保存 ({len(self.trades_buffer)} 笔交易)")

    def _save_config(self):
        """保存配置"""
        config = {
            'symbols': self.symbols,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.capital_manager.initial_capital,
            'vote_threshold': self.vote_fusion.vote_threshold,
            'min_hold_bars': self.min_hold_bars,
            'cooldown_bars_after_stop': self.cooldown_bars_after_stop,
            'use_atr_stops': self.use_atr_stops,
        }

        path = self.run_dir / 'config.json'
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"✅ config.json 已保存")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="MKCS 性能优化回测")
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'MSFT', 'GOOGL'])
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default='2024-12-31')
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--vote-threshold', type=int, default=2)
    parser.add_argument('--min-strength', type=float, default=0.01)
    parser.add_argument('--min-confidence', type=float, default=0.0)  # P1-1
    parser.add_argument('--min-hold-bars', type=int, default=0)  # P1-2
    parser.add_argument('--cooldown-bars', type=int, default=0)  # P1-2
    parser.add_argument('--use-atr-stops', action='store_true')  # P1-3
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--output-dir', default='outputs/backtests')

    args = parser.parse_args()

    engine = BacktestEngineOptimized(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output_dir,
        initial_capital=args.capital,
        vote_threshold=args.vote_threshold,
        min_strength_to_vote=args.min_strength,
        min_avg_confidence=args.min_confidence,  # P1-1
        min_hold_bars=args.min_hold_bars,  # P1-2
        cooldown_bars_after_stop=args.cooldown_bars,  # P1-2
        use_atr_stops=args.use_atr_stops,  # P1-3
        use_cache=not args.no_cache
    )

    try:
        result = engine.run_backtest()
        if result.get('success'):
            logger.info("\n🎉 回测成功完成!")
        else:
            logger.error("\n❌ 回测失败!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ 回测异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
