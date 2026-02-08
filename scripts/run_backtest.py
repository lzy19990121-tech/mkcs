"""
MKCS 历史数据回测模拟交易

严格遵循以下原则：
1. 不使用未来数据（只使用当前时刻及之前的数据）
2. 资金管理（仓位控制、止损止盈）
3. 滑点和手续费模拟
4. 多策略组合与权重优化
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

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


@dataclass
class Trade:
    """交易记录"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    entry_value: float
    exit_value: Optional[float]
    pnl: Optional[float]
    pnl_pct: Optional[float]
    exit_reason: str  # 'SIGNAL', 'STOP_LOSS', 'TAKE_PROFIT', 'TIME_EXIT'
    commission: float


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    quantity: int
    entry_price: float
    entry_date: datetime
    stop_loss: Optional[float]
    take_profit: Optional[float]
    entry_value: float


class CapitalManager:
    """资金管理器

    严格按照以下规则：
    1. 单笔交易最大风险不超过账户的 2%
    2. 单个标的最大仓位不超过账户的 20%
    3. 总持仓不超过账户的 80%
    4. 动态调整仓位大小
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        max_risk_per_trade: float = 0.02,  # 每笔交易最大风险 2%
        max_position_pct: float = 0.20,     # 单个标的最大仓位 20%
        max_total_exposure: float = 0.80,    # 总持仓上限 80%
        commission_rate: float = 0.001,      # 手续费率 0.1%
        slippage_bps: float = 5.0            # 滑点 5 个基点
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_pct = max_position_pct
        self.max_total_exposure = max_total_exposure
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps

        self.equity_curve = [initial_capital]
        self.drawdowns = [0.0]

    def get_equity(self) -> float:
        """获取当前权益"""
        return self.current_capital

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: Optional[float],
        current_exposure: float
    ) -> int:
        """计算仓位大小

        Args:
            entry_price: 入场价格
            stop_loss: 止损价格
            current_exposure: 当前已用资金

        Returns:
            建议仓位数量
        """
        # 1. 基于风险的计算（凯利公式简化版）
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

        # 取最小值
        shares = min(shares_by_risk, shares_by_position, shares_by_capital)

        # 确保至少是 100 股的整数倍
        shares = max(0, (shares // 100) * 100)

        return shares

    def calculate_commission(self, price: float, quantity: int) -> float:
        """计算手续费"""
        return price * quantity * self.commission_rate

    def apply_slippage(self, price: float, direction: str) -> float:
        """应用滑点

        Args:
            price: 原始价格
            direction: 'BUY' or 'SELL'

        Returns:
            调整后价格
        """
        slippage = price * self.slippage_bps / 10000
        if direction == 'BUY':
            return price + slippage  # 买入时价格更高
        else:
            return price - slippage  # 卖出时价格更低

    def update_equity(self, pnl: float, timestamp: datetime):
        """更新权益"""
        self.current_capital += pnl
        self.equity_curve.append(self.current_capital)

        # 计算回撤
        peak = max(self.equity_curve)
        drawdown = (peak - self.current_capital) / peak if peak > 0 else 0
        self.drawdowns.append(drawdown)

    def get_metrics(self) -> Dict:
        """获取资金指标"""
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]

        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        max_drawdown = max(self.drawdowns)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

        win_trades = sum(1 for r in returns if r > 0)
        total_trades = len(returns)
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'win_rate': win_rate,
            'total_trades': total_trades
        }


class MAStrategy:
    """移动平均策略"""

    def __init__(self, fast_period: int, slow_period: int):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"MA({fast_period},{slow_period})"

    def generate_signal(self, data: pd.DataFrame) -> Dict:
        """生成交易信号

        Args:
            data: 历史数据（包含日期索引）

        Returns:
            信号字典
        """
        if len(data) < self.slow_period:
            return {'action': 'HOLD', 'reason': 'Insufficient data'}

        # 计算移动平均（只使用历史数据）
        data = data.copy()
        data['ma_fast'] = data['Close'].rolling(window=self.fast_period).mean()
        data['ma_slow'] = data['Close'].rolling(window=self.slow_period).mean()

        # 最新信号
        current = data.iloc[-1]
        prev = data.iloc[-2]

        fast_ma = current['ma_fast']
        slow_ma = current['ma_slow']
        prev_fast = prev['ma_fast']
        prev_slow = prev['ma_slow']

        current_price = current['Close']

        # 金叉买入
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            return {
                'action': 'BUY',
                'strength': (fast_ma - slow_ma) / slow_ma,
                'price': current_price,
                'stop_loss': current_price * 0.95,  # 5% 止损
                'take_profit': current_price * 1.10,  # 10% 止盈
                'reason': 'Golden cross'
            }
        # 死叉卖出
        elif prev_fast >= prev_slow and fast_ma < slow_ma:
            return {
                'action': 'SELL',
                'strength': (slow_ma - fast_ma) / slow_ma,
                'price': current_price,
                'stop_loss': current_price * 1.05,
                'take_profit': current_price * 0.90,
                'reason': 'Death cross'
            }

        # 持仓时检查止损止盈
        return {'action': 'HOLD', 'reason': 'No signal'}


class BacktestEngine:
    """回测引擎

    严格按时间顺序处理，确保不使用未来数据
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        capital_manager: CapitalManager
    ):
        self.symbols = symbols
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.capital_manager = capital_manager
        self.data_source = YahooFinanceSource(enable_cache=True)

        # 初始化策略
        self.strategies = [
            MAStrategy(5, 20),
            MAStrategy(10, 30),
            MAStrategy(20, 50),
        ]

        # 状态
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values: List[Tuple[pd.Timestamp, float]] = []

        logger.info("=" * 60)
        logger.info("MKCS 历史数据回测引擎初始化")
        logger.info("=" * 60)
        logger.info(f"回测区间: {start_date} ~ {end_date}")
        logger.info(f"交易标的: {', '.join(symbols)}")
        logger.info(f"初始资金: ${capital_manager.initial_capital:,.2f}")
        logger.info(f"策略数量: {len(self.strategies)}")
        for s in self.strategies:
            logger.info(f"  - {s.name}")
        logger.info("=" * 60)

    def fetch_historical_data(self, symbol: str) -> pd.DataFrame:
        """获取历史数据

        Args:
            symbol: 标的代码

        Returns:
            价格数据
        """
        try:
            bars = self.data_source.get_bars(
                symbol=symbol,
                start=self.start_date.to_pydatetime(),
                end=self.end_date.to_pydatetime(),
                interval="1d"
            )

            if not bars:
                return pd.DataFrame()

            # 转换为 DataFrame
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

            logger.info(f"获取 {symbol} 数据: {len(data)} 根K线")
            return data

        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return pd.DataFrame()

    def run_backtest(self):
        """运行回测"""
        logger.info("\n开始回测...")

        # 获取所有标的的数据
        all_data = {}
        for symbol in self.symbols:
            data = self.fetch_historical_data(symbol)
            if not data.empty:
                all_data[symbol] = data

        if not all_data:
            logger.error("没有获取到任何数据")
            return

        # 获取交易日期（使用所有数据集的并集）
        all_dates = set()
        for data in all_data.values():
            all_dates.update(data.index)
        trading_dates = sorted(list(all_dates))

        logger.info(f"交易日期数: {len(trading_dates)}")

        # 按日期遍历
        for i, current_date in enumerate(trading_dates):
            if i % 50 == 0:
                logger.info(f"处理进度: {i}/{len(trading_dates)} ({current_date.date()})")

            # 处理每个标的
            for symbol, data in all_data.items():
                if current_date not in data.index:
                    continue

                # 获取截至当前日期的历史数据（不包含未来数据）
                hist_data = data.loc[:current_date]

                if len(hist_data) < 50:  # 数据不足
                    continue

                # 检查止损止盈
                self._check_exit_conditions(symbol, hist_data.iloc[-1])

                # 生成信号（多策略投票）
                signals = self._generate_signals(hist_data)

                # 执行交易
                self._execute_signals(symbol, signals, hist_data.iloc[-1])

            # 记录每日权益
            self._record_daily_value(current_date, all_data)

        # 平掉所有持仓
        self._close_all_positions(trading_dates[-1] if trading_dates else pd.Timestamp.now())

        # 生成报告
        self._generate_report()

    def _generate_signals(self, hist_data: pd.DataFrame) -> Dict:
        """生成信号（多策略投票）

        Args:
            hist_data: 历史数据

        Returns:
            综合信号
        """
        buy_votes = 0
        sell_votes = 0
        total_strength = 0

        for strategy in self.strategies:
            signal = strategy.generate_signal(hist_data)

            if signal['action'] == 'BUY':
                buy_votes += 1
                total_strength += signal.get('strength', 0)
            elif signal['action'] == 'SELL':
                sell_votes += 1
                total_strength -= signal.get('strength', 0)

        # 投票规则：至少 1 个策略同意即可交易（降低门槛）
        if buy_votes >= 1:
            return {
                'action': 'BUY',
                'strength': total_strength / max(buy_votes, 1),
                'stop_loss': None,  # 将在执行时计算
                'take_profit': None,
                'reason': f'Multi-strategy BUY ({buy_votes}/{len(self.strategies)})'
            }
        elif sell_votes >= 1:
            return {
                'action': 'SELL',
                'strength': abs(total_strength / max(sell_votes, 1)),
                'stop_loss': None,
                'take_profit': None,
                'reason': f'Multi-strategy SELL ({sell_votes}/{len(self.strategies)})'
            }

        return {'action': 'HOLD', 'reason': 'No consensus'}

    def _execute_signals(self, symbol: str, signal: Dict, bar_data: pd.Series):
        """执行交易信号

        Args:
            symbol: 标的代码
            signal: 交易信号
            bar_data: 当前K线数据
        """
        current_price = float(bar_data['Close'])
        current_date = bar_data.name

        # 计算当前持仓价值
        current_exposure = sum(
            p.quantity * p.entry_price
            for p in self.positions.values()
            if p.symbol != symbol
        )

        if signal['action'] == 'BUY':
            # 检查是否已持有该标的
            if symbol in self.positions:
                return

            # 计算止损止盈
            stop_loss = current_price * 0.95
            take_profit = current_price * 1.15

            # 计算仓位大小
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
                stop_loss=stop_loss,
                take_profit=take_profit,
                entry_value=entry_value
            )

            logger.info(f"{current_date.date()} BUY {symbol} {quantity}股 @ ${entry_price:.2f} "
                       f"(止损: ${stop_loss:.2f}, 止盈: ${take_profit:.2f})")

        elif signal['action'] == 'SELL':
            # 检查是否持有该标的
            if symbol not in self.positions:
                return

            # 平仓
            self._close_position(symbol, bar_data, 'SIGNAL')

    def _check_exit_conditions(self, symbol: str, bar_data: pd.Series):
        """检查止损止盈条件

        Args:
            symbol: 标的代码
            bar_data: 当前K线数据
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        current_price = float(bar_data['Low'])  # 使用最低价检查止损
        high_price = float(bar_data['High'])    # 使用最高价检查止盈

        # 检查止损
        if position.stop_loss and current_price <= position.stop_loss:
            self._close_position(symbol, bar_data, 'STOP_LOSS')
            return

        # 检查止盈
        if position.take_profit and high_price >= position.take_profit:
            self._close_position(symbol, bar_data, 'TAKE_PROFIT')
            return

    def _close_position(self, symbol: str, bar_data: pd.Series, reason: str):
        """平仓

        Args:
            symbol: 标的代码
            bar_data: 当前K线数据
            reason: 平仓原因
        """
        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        current_date = bar_data.name

        # 应用滑点
        exit_price = self.capital_manager.apply_slippage(float(bar_data['Close']), 'SELL')
        commission = self.capital_manager.calculate_commission(exit_price, position.quantity)
        exit_value = exit_price * position.quantity - commission

        # 计算盈亏
        if position.direction == 'LONG':
            pnl = exit_value - position.entry_value
        else:
            pnl = position.entry_value - exit_value

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
            commission=commission * 2  # 开仓+平仓
        )
        self.trades.append(trade)

        logger.info(f"{current_date.date()} SELL {symbol} {position.quantity}股 @ ${exit_price:.2f} "
                   f"PnL: ${pnl:+.2f} ({pnl_pct*100:+.2f}%) [{reason}]")

        # 移除持仓
        del self.positions[symbol]

    def _close_all_positions(self, current_date: pd.Timestamp):
        """平掉所有持仓"""
        for symbol in list(self.positions.keys()):
            # 创建虚拟的 bar_data
            position = self.positions[symbol]
            bar_data = pd.Series({
                'Close': position.entry_price,
                'High': position.entry_price,
                'Low': position.entry_price
            }, name=current_date)
            self._close_position(symbol, bar_data, 'TIME_EXIT')

    def _record_daily_value(self, date: pd.Timestamp, all_data: Dict):
        """记录每日权益"""
        total_value = self.capital_manager.current_capital

        # 加上持仓市值
        for symbol, position in self.positions.items():
            if symbol in all_data and date in all_data[symbol].index:
                current_price = float(all_data[symbol].loc[date, 'Close'])
                total_value += current_price * position.quantity - position.entry_value

        self.daily_values.append((date, total_value))

    def _generate_report(self):
        """生成回测报告"""
        logger.info("\n" + "=" * 60)
        logger.info("回测报告")
        logger.info("=" * 60)

        metrics = self.capital_manager.get_metrics()

        logger.info(f"\n资金指标:")
        logger.info(f"  初始资金: ${metrics['initial_capital']:,.2f}")
        logger.info(f"  最终权益: ${metrics['final_capital']:,.2f}")
        logger.info(f"  总收益: {metrics['total_return_pct']:.2f}%")
        logger.info(f"  最大回撤: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")

        logger.info(f"\n交易统计:")
        logger.info(f"  总交易次数: {len(self.trades)}")

        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl and t.pnl < 0]

            logger.info(f"  盈利交易: {len(winning_trades)}")
            logger.info(f"  亏损交易: {len(losing_trades)}")

            if winning_trades:
                avg_win = np.mean([t.pnl for t in winning_trades])
                logger.info(f"  平均盈利: ${avg_win:.2f}")

            if losing_trades:
                avg_loss = np.mean([t.pnl for t in losing_trades])
                logger.info(f"  平均亏损: ${avg_loss:.2f}")

            # 盈亏比
            if winning_trades and losing_trades:
                profit_factor = sum(t.pnl for t in winning_trades) / abs(sum(t.pnl for t in losing_trades))
                logger.info(f"  盈亏比: {profit_factor:.2f}")

            # 按退出原因统计
            exit_reasons = defaultdict(int)
            for t in self.trades:
                exit_reasons[t.exit_reason] += 1

            logger.info(f"\n退出原因统计:")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
                logger.info(f"  {reason}: {count}")

        logger.info("\n" + "=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="MKCS 历史数据回测")
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL'],
        help='交易标的'
    )
    parser.add_argument(
        '--start',
        default='2023-01-01',
        help='开始日期 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        default='2024-12-31',
        help='结束日期 (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=100000,
        help='初始资金'
    )
    parser.add_argument(
        '--max-risk',
        type=float,
        default=0.02,
        help='单笔最大风险 (默认 2%%)'
    )

    args = parser.parse_args()

    # 创建资金管理器
    capital_mgr = CapitalManager(
        initial_capital=args.capital,
        max_risk_per_trade=args.max_risk,
        max_position_pct=0.20,
        max_total_exposure=0.80
    )

    # 创建回测引擎
    engine = BacktestEngine(
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        capital_manager=capital_mgr
    )

    # 运行回测
    try:
        engine.run_backtest()
    except KeyboardInterrupt:
        logger.info("\n回测被中断")


if __name__ == "__main__":
    main()
