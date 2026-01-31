"""
回测性能分析指标

计算各种风险调整收益指标和交易统计
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional, Tuple
import math
from collections import defaultdict

from core.models import Trade, Position


@dataclass(frozen=True)
class PerformanceMetrics:
    """性能指标数据类"""

    # 基本收益指标
    total_return: Decimal           # 总收益率
    annualized_return: Decimal      # 年化收益率
    daily_returns: List[Decimal]    # 每日收益率列表

    # 风险指标
    volatility: Decimal             # 年化波动率
    max_drawdown: Decimal           # 最大回撤
    max_drawdown_duration: int      # 最大回撤持续时间（天）
    sharpe_ratio: Decimal           # 夏普比率
    sortino_ratio: Decimal          # 索提诺比率
    calmar_ratio: Decimal           # 卡尔马比率

    # 交易统计
    total_trades: int               # 总交易次数
    winning_trades: int             # 盈利交易次数
    losing_trades: int              # 亏损交易次数
    win_rate: Decimal               # 胜率
    profit_factor: Decimal          # 盈亏比
    average_profit: Decimal         # 平均盈利
    average_loss: Decimal           # 平均亏损
    largest_profit: Decimal         # 最大单笔盈利
    largest_loss: Decimal           # 最大单笔亏损
    average_trade_return: Decimal   # 平均每笔交易收益

    # 持仓指标
    avg_holding_period: float       # 平均持仓周期（天）
    max_consecutive_wins: int       # 最大连续盈利次数
    max_consecutive_losses: int     # 最大连续亏损次数

    # 基准对比
    alpha: Optional[Decimal]        # 阿尔法
    beta: Optional[Decimal]         # 贝塔
    information_ratio: Optional[Decimal]  # 信息比率

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "total_return": float(self.total_return),
            "annualized_return": float(self.annualized_return),
            "volatility": float(self.volatility),
            "max_drawdown": float(self.max_drawdown),
            "max_drawdown_duration": self.max_drawdown_duration,
            "sharpe_ratio": float(self.sharpe_ratio),
            "sortino_ratio": float(self.sortino_ratio),
            "calmar_ratio": float(self.calmar_ratio),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": float(self.win_rate),
            "profit_factor": float(self.profit_factor),
            "average_profit": float(self.average_profit),
            "average_loss": float(self.average_loss),
            "largest_profit": float(self.largest_profit),
            "largest_loss": float(self.largest_loss),
            "average_trade_return": float(self.average_trade_return),
            "avg_holding_period": self.avg_holding_period,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "alpha": float(self.alpha) if self.alpha else None,
            "beta": float(self.beta) if self.beta else None,
            "information_ratio": float(self.information_ratio) if self.information_ratio else None,
        }


class MetricsCalculator:
    """性能指标计算器"""

    # 无风险利率（年化，假设 2%）
    RISK_FREE_RATE = Decimal("0.02")

    @classmethod
    def calculate(
        cls,
        equity_curve: List[Tuple[datetime, Decimal]],
        trades: List[Trade],
        initial_cash: Decimal,
        benchmark_returns: Optional[List[Decimal]] = None
    ) -> PerformanceMetrics:
        """计算完整的性能指标

        Args:
            equity_curve: 权益曲线 [(timestamp, equity), ...]
            trades: 交易记录列表
            initial_cash: 初始资金
            benchmark_returns: 基准收益率列表（可选）

        Returns:
            PerformanceMetrics 对象
        """
        if not equity_curve:
            raise ValueError("权益曲线不能为空")

        # 计算每日收益率
        daily_returns = cls._calculate_daily_returns(equity_curve)

        # 基本收益指标
        final_equity = equity_curve[-1][1]
        total_return = (final_equity - initial_cash) / initial_cash

        # 年化收益率
        days = (equity_curve[-1][0] - equity_curve[0][0]).days
        years = max(days / 365.0, 0.01)
        annualized_return = ((final_equity / initial_cash) ** (Decimal("1") / Decimal(str(years)))) - Decimal("1")

        # 风险指标
        volatility = cls._calculate_volatility(daily_returns)
        max_drawdown, max_dd_duration = cls._calculate_drawdown(equity_curve)

        # 风险调整收益指标
        sharpe_ratio = cls._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = cls._calculate_sortino_ratio(daily_returns)
        calmar_ratio = cls._calculate_calmar_ratio(annualized_return, max_drawdown)

        # 交易统计
        trade_stats = cls._calculate_trade_stats(trades)

        # 基准对比
        alpha, beta, info_ratio = None, None, None
        if benchmark_returns and len(benchmark_returns) == len(daily_returns):
            alpha, beta = cls._calculate_alpha_beta(daily_returns, benchmark_returns)
            info_ratio = cls._calculate_information_ratio(daily_returns, benchmark_returns)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            daily_returns=daily_returns,
            volatility=volatility,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            average_profit=trade_stats["average_profit"],
            average_loss=trade_stats["average_loss"],
            largest_profit=trade_stats["largest_profit"],
            largest_loss=trade_stats["largest_loss"],
            average_trade_return=trade_stats["average_trade_return"],
            avg_holding_period=trade_stats["avg_holding_period"],
            max_consecutive_wins=trade_stats["max_consecutive_wins"],
            max_consecutive_losses=trade_stats["max_consecutive_losses"],
            alpha=alpha,
            beta=beta,
            information_ratio=info_ratio
        )

    @classmethod
    def _calculate_daily_returns(
        cls,
        equity_curve: List[Tuple[datetime, Decimal]]
    ) -> List[Decimal]:
        """计算每日收益率"""
        # 按日期分组，取每天的最后一个权益值
        daily_equity = {}
        for ts, equity in equity_curve:
            date = ts.date()
            daily_equity[date] = equity

        dates = sorted(daily_equity.keys())
        returns = []

        for i in range(1, len(dates)):
            prev_equity = daily_equity[dates[i - 1]]
            curr_equity = daily_equity[dates[i]]
            daily_return = (curr_equity - prev_equity) / prev_equity
            returns.append(daily_return)

        return returns

    @classmethod
    def _calculate_volatility(
        cls,
        daily_returns: List[Decimal]
    ) -> Decimal:
        """计算年化波动率"""
        if len(daily_returns) < 2:
            return Decimal("0")

        # 计算标准差
        mean_return = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_return) ** 2 for r in daily_returns) / len(daily_returns)
        std_dev = Decimal(str(math.sqrt(float(variance))))

        # 年化（假设252个交易日）
        return std_dev * Decimal(str(math.sqrt(252)))

    @classmethod
    def _calculate_drawdown(
        cls,
        equity_curve: List[Tuple[datetime, Decimal]]
    ) -> Tuple[Decimal, int]:
        """计算最大回撤和持续时间

        Returns:
            (最大回撤, 最大回撤持续天数)
        """
        max_drawdown = Decimal("0")
        max_dd_duration = 0

        peak = equity_curve[0][1]
        peak_date = equity_curve[0][0]

        for ts, equity in equity_curve:
            if equity > peak:
                peak = equity
                peak_date = ts
            else:
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    max_dd_duration = (ts - peak_date).days

        return max_drawdown, max_dd_duration

    @classmethod
    def _calculate_sharpe_ratio(
        cls,
        daily_returns: List[Decimal]
    ) -> Decimal:
        """计算夏普比率"""
        if len(daily_returns) < 2:
            return Decimal("0")

        # 年化收益率
        avg_daily_return = sum(daily_returns) / len(daily_returns)
        annual_return = avg_daily_return * 252

        # 年化波动率
        variance = sum((r - avg_daily_return) ** 2 for r in daily_returns) / len(daily_returns)
        annual_volatility = Decimal(str(math.sqrt(float(variance) * 252)))

        if annual_volatility == 0:
            return Decimal("0")

        return (annual_return - cls.RISK_FREE_RATE) / annual_volatility

    @classmethod
    def _calculate_sortino_ratio(
        cls,
        daily_returns: List[Decimal]
    ) -> Decimal:
        """计算索提诺比率（只考虑下行波动）"""
        if len(daily_returns) < 2:
            return Decimal("0")

        avg_daily_return = sum(daily_returns) / len(daily_returns)
        annual_return = avg_daily_return * 252

        # 下行标准差（只考虑负收益）
        negative_returns = [r for r in daily_returns if r < 0]
        if not negative_returns:
            return Decimal("0")

        downside_variance = sum(r ** 2 for r in negative_returns) / len(negative_returns)
        downside_deviation = Decimal(str(math.sqrt(float(downside_variance) * 252)))

        if downside_deviation == 0:
            return Decimal("0")

        return (annual_return - cls.RISK_FREE_RATE) / downside_deviation

    @classmethod
    def _calculate_calmar_ratio(
        cls,
        annualized_return: Decimal,
        max_drawdown: Decimal
    ) -> Decimal:
        """计算卡尔马比率"""
        if max_drawdown == 0:
            return Decimal("0")
        return annualized_return / max_drawdown

    @classmethod
    def _calculate_trade_stats(
        cls,
        trades: List[Trade]
    ) -> Dict:
        """计算交易统计"""
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": Decimal("0"),
                "profit_factor": Decimal("0"),
                "average_profit": Decimal("0"),
                "average_loss": Decimal("0"),
                "largest_profit": Decimal("0"),
                "largest_loss": Decimal("0"),
                "average_trade_return": Decimal("0"),
                "avg_holding_period": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
            }

        # 按 symbol 和方向计算每笔交易的盈亏
        trade_pnls = []
        holding_periods = []

        for trade in trades:
            # 简化计算：假设所有交易都已平仓
            # 实际应该配对买卖计算
            realized_pnl = trade.realized_pnl if hasattr(trade, 'realized_pnl') else Decimal("0")
            trade_pnls.append(realized_pnl)

            # 计算持仓时间（如果有字段）
            if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
                holding_days = (trade.exit_time - trade.entry_time).days
                holding_periods.append(holding_days)

        # 盈利和亏损交易
        profits = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]

        total_trades = len(trade_pnls)
        winning_trades = len(profits)
        losing_trades = len(losses)

        win_rate = Decimal(str(winning_trades / total_trades)) if total_trades > 0 else Decimal("0")

        # 盈亏比
        total_profit = sum(profits) if profits else Decimal("0")
        total_loss = abs(sum(losses)) if losses else Decimal("0")
        profit_factor = total_profit / total_loss if total_loss > 0 else Decimal("0")

        # 平均盈亏
        average_profit = total_profit / winning_trades if winning_trades > 0 else Decimal("0")
        average_loss = -total_loss / losing_trades if losing_trades > 0 else Decimal("0")

        # 最大单笔盈亏
        largest_profit = max(profits) if profits else Decimal("0")
        largest_loss = min(losses) if losses else Decimal("0")

        # 平均交易收益
        total_pnl = sum(trade_pnls)
        average_trade_return = total_pnl / total_trades if total_trades > 0 else Decimal("0")

        # 平均持仓周期
        avg_holding_period = sum(holding_periods) / len(holding_periods) if holding_periods else 0.0

        # 连续盈亏次数
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in trade_pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_profit": average_profit,
            "average_loss": average_loss,
            "largest_profit": largest_profit,
            "largest_loss": largest_loss,
            "average_trade_return": average_trade_return,
            "avg_holding_period": avg_holding_period,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
        }

    @classmethod
    def _calculate_alpha_beta(
        cls,
        returns: List[Decimal],
        benchmark_returns: List[Decimal]
    ) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """计算阿尔法和贝塔"""
        if len(returns) < 2 or len(returns) != len(benchmark_returns):
            return None, None

        # 计算平均收益
        avg_return = sum(returns) / len(returns)
        avg_benchmark = sum(benchmark_returns) / len(benchmark_returns)

        # 计算协方差和方差
        covariance = sum(
            (r - avg_return) * (b - avg_benchmark)
            for r, b in zip(returns, benchmark_returns)
        ) / len(returns)

        benchmark_variance = sum(
            (b - avg_benchmark) ** 2
            for b in benchmark_returns
        ) / len(benchmark_returns)

        if benchmark_variance == 0:
            return None, None

        # 贝塔 = 协方差 / 基准方差
        beta = covariance / benchmark_variance

        # 阿尔法 = 实际收益 - (无风险利率 + 贝塔 * (基准收益 - 无风险利率))
        alpha = avg_return - (cls.RISK_FREE_RATE / 252 + beta * (avg_benchmark - cls.RISK_FREE_RATE / 252))

        # 年化阿尔法
        alpha = alpha * 252

        return alpha, beta

    @classmethod
    def _calculate_information_ratio(
        cls,
        returns: List[Decimal],
        benchmark_returns: List[Decimal]
    ) -> Optional[Decimal]:
        """计算信息比率"""
        if len(returns) < 2 or len(returns) != len(benchmark_returns):
            return None

        # 超额收益
        excess_returns = [r - b for r, b in zip(returns, benchmark_returns)]
        avg_excess = sum(excess_returns) / len(excess_returns)

        # 跟踪误差（超额收益的标准差）
        variance = sum((er - avg_excess) ** 2 for er in excess_returns) / len(excess_returns)
        tracking_error = Decimal(str(math.sqrt(float(variance))))

        if tracking_error == 0:
            return None

        # 年化
        return avg_excess * 252 / tracking_error


class MetricsReport:
    """性能指标报告生成器"""

    @classmethod
    def generate_report(
        cls,
        metrics: PerformanceMetrics,
        title: str = "回测性能报告"
    ) -> str:
        """生成格式化的性能报告"""
        lines = [
            f"# {title}",
            "",
            "## 收益指标",
            f"- **总收益率**: {metrics.total_return*100:.2f}%",
            f"- **年化收益率**: {metrics.annualized_return*100:.2f}%",
            "",
            "## 风险指标",
            f"- **年化波动率**: {metrics.volatility*100:.2f}%",
            f"- **最大回撤**: {metrics.max_drawdown*100:.2f}%",
            f"- **最大回撤持续时间**: {metrics.max_drawdown_duration} 天",
            "",
            "## 风险调整收益",
            f"- **夏普比率**: {metrics.sharpe_ratio:.2f}",
            f"- **索提诺比率**: {metrics.sortino_ratio:.2f}",
            f"- **卡尔马比率**: {metrics.calmar_ratio:.2f}",
            "",
            "## 交易统计",
            f"- **总交易次数**: {metrics.total_trades}",
            f"- **盈利次数**: {metrics.winning_trades}",
            f"- **亏损次数**: {metrics.losing_trades}",
            f"- **胜率**: {metrics.win_rate*100:.2f}%",
            f"- **盈亏比**: {metrics.profit_factor:.2f}",
            f"- **平均盈利**: ${metrics.average_profit:.2f}",
            f"- **平均亏损**: ${metrics.average_loss:.2f}",
            f"- **最大单笔盈利**: ${metrics.largest_profit:.2f}",
            f"- **最大单笔亏损**: ${metrics.largest_loss:.2f}",
            "",
            "## 其他指标",
            f"- **平均持仓周期**: {metrics.avg_holding_period:.1f} 天",
            f"- **最大连续盈利**: {metrics.max_consecutive_wins} 次",
            f"- **最大连续亏损**: {metrics.max_consecutive_losses} 次",
        ]

        if metrics.alpha is not None and metrics.beta is not None:
            lines.extend([
                "",
                "## 基准对比",
                f"- **阿尔法**: {metrics.alpha*100:.2f}%",
                f"- **贝塔**: {metrics.beta:.2f}",
                f"- **信息比率**: {metrics.information_ratio:.2f}" if metrics.information_ratio else "",
            ])

        return "\n".join(lines)


if __name__ == "__main__":
    """自测代码"""
    print("=== 性能指标计算器测试 ===\n")

    from datetime import timedelta
    from core.models import Trade, TradeSide

    # 构造测试数据
    initial_cash = Decimal("100000")

    # 模拟权益曲线
    equity_curve = []
    current_equity = initial_cash
    start_date = datetime(2024, 1, 1)

    for i in range(30):  # 30天
        date = start_date + timedelta(days=i)
        # 模拟有涨有跌
        change = Decimal(str(0.005 * (1 if i % 3 != 0 else -1)))
        current_equity = current_equity * (Decimal("1") + change)
        equity_curve.append((date, current_equity))

    # 模拟交易
    trades = []
    for i in range(10):
        is_win = i % 3 != 0  # 70% 胜率
        pnl = Decimal("500") if is_win else Decimal("-200")
        trade = Trade(
            trade_id=f"T{i}",
            symbol="AAPL",
            side=TradeSide.BUY if i % 2 == 0 else TradeSide.SELL,
            quantity=100,
            price=Decimal("150"),
            timestamp=start_date + timedelta(days=i),
            commission=Decimal("1"),
            realized_pnl=pnl
        )
        trades.append(trade)

    # 计算指标
    print("计算性能指标...")
    metrics = MetricsCalculator.calculate(
        equity_curve=equity_curve,
        trades=trades,
        initial_cash=initial_cash
    )

    # 打印报告
    report = MetricsReport.generate_report(metrics, "测试回测报告")
    print(report)

    print("\n✓ 测试完成")
