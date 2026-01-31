"""
交易Agent运行器

负责任务编排和状态管理，协调各个skill完成回测
"""

from datetime import datetime, date, timedelta
from decimal import Decimal
from typing import List, Optional, Dict, Any
import sys
import argparse
import csv
import json
from pathlib import Path

from core.context import RunContext
from core.models import Position, OrderIntent, Signal
from skills.market_data.mock_source import MarketDataSource, MockMarketSource
from skills.market_data.csv_source import CSVDataSource
from skills.strategy.base import Strategy
from skills.strategy.moving_average import MAStrategy
from skills.risk.basic_risk import RiskManager, BasicRiskManager
from skills.session.trading_session import TradingSession
from broker.paper import PaperBroker
from storage.db import TradeDB
from events.event_log import EventLogger, get_event_logger
from agent.replay_engine import ReplayEngine, ReplayMode
from config import BacktestConfig


class TradingAgent:
    """自动交易Agent

    职责：
    1. 编排各个skill
    2. 执行回测主循环
    3. 记录交易数据
    """

    def __init__(
        self,
        data_source: MarketDataSource,
        strategy: Strategy,
        risk_manager: RiskManager,
        broker: PaperBroker,
        db: Optional[TradeDB] = None,
        event_logger: Optional[EventLogger] = None,
        config: Optional[BacktestConfig] = None
    ):
        """初始化Agent

        Args:
            data_source: 市场数据源
            strategy: 交易策略
            risk_manager: 风控管理器
            broker: 模拟经纪商
            db: 数据库（可选）
            event_logger: 事件日志器（可选）
            config: 回测配置（可选）
        """
        self.data_source = data_source
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.broker = broker
        self.db = db
        self.event_logger = event_logger or get_event_logger()
        self.config = config

        # 初始化风控管理器的资金
        self.risk_manager.set_capital(float(self.broker.initial_cash))

    def tick(self, ctx: RunContext, symbols: List[str]) -> dict:
        """单次时间推进"""
        result = {
            "date": ctx.trading_date,
            "symbols_processed": 0,
            "signals_generated": 0,
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
            "risk_rejects": []
        }

        for symbol in symbols:
            result["symbols_processed"] += 1

            bars = self.data_source.get_bars_until(symbol, ctx.now, ctx.bar_interval)
            start_dt = ctx.now - timedelta(days=self.strategy.get_min_bars_required() + 10)
            bars = [b for b in bars if b.timestamp >= start_dt]

            if len(bars) < self.strategy.get_min_bars_required():
                continue

            self.event_logger.log_event(
                ts=ctx.now,
                symbol=symbol,
                stage="data_fetch",
                payload={"bars_count": len(bars), "interval": ctx.bar_interval},
                reason=f"fetch_bars_until {ctx.now.isoformat()}"
            )

            position = self.broker.get_position(symbol)
            signals = self.strategy.generate_signals(bars, position)

            for signal in signals:
                result["signals_generated"] += 1

                self.event_logger.log_event(
                    ts=ctx.now,
                    symbol=symbol,
                    stage="signal_gen",
                    payload={
                        "action": signal.action,
                        "price": float(signal.price),
                        "quantity": signal.quantity,
                        "confidence": signal.confidence
                    },
                    reason=signal.reason
                )

                intent = self.risk_manager.check(
                    signal,
                    self.broker.get_positions(),
                    float(self.broker.get_cash_balance())
                )

                self.event_logger.log_event(
                    ts=ctx.now,
                    symbol=symbol,
                    stage="risk_check",
                    payload={
                        "approved": intent.approved,
                        "action": intent.signal.action
                    },
                    reason=intent.risk_reason
                )

                if not intent.approved:
                    result["orders_rejected"] += 1
                    result["risk_rejects"].append({
                        "date": ctx.trading_date,
                        "symbol": symbol,
                        "action": intent.signal.action,
                        "reason": intent.risk_reason
                    })
                    self.event_logger.log_event(
                        ts=ctx.now,
                        symbol=symbol,
                        stage="order_reject",
                        payload={
                            "action": intent.signal.action,
                            "quantity": intent.signal.quantity
                        },
                        reason=intent.risk_reason
                    )
                    continue

                order = self.broker.submit_order(intent)
                if order:
                    result["orders_submitted"] += 1
                    self.event_logger.log_event(
                        ts=ctx.now,
                        symbol=symbol,
                        stage="order_submit",
                        payload={
                            "order_id": order.order_id,
                            "side": order.side,
                            "quantity": order.quantity
                        },
                        reason="submit_market_order"
                    )
                else:
                    result["orders_rejected"] += 1
                    self.event_logger.log_event(
                        ts=ctx.now,
                        symbol=symbol,
                        stage="order_reject",
                        payload={
                            "action": intent.signal.action,
                            "quantity": intent.signal.quantity
                        },
                        reason="submit_failed"
                    )

            current_bar = bars[-1]
            fills, rejects = self.broker.on_bar(current_bar)

            for fill in fills:
                trade = fill.trade
                result["orders_filled"] += 1
                self.event_logger.log_event(
                    ts=trade.timestamp,
                    symbol=trade.symbol,
                    stage="order_fill",
                    payload={
                        "order_id": fill.order.order_id,
                        "trade_id": trade.trade_id,
                        "side": trade.side,
                        "price": float(trade.price),
                        "quantity": trade.quantity,
                        "commission": float(trade.commission)
                    },
                    reason="filled_on_next_bar_open"
                )
                if self.db:
                    self.db.save_trade(trade)

            for reject in rejects:
                result["orders_rejected"] += 1
                self.event_logger.log_event(
                    ts=ctx.now,
                    symbol=reject.order.symbol,
                    stage="order_reject",
                    payload={
                        "order_id": reject.order.order_id,
                        "side": reject.order.side,
                        "quantity": reject.order.quantity
                    },
                    reason=reject.reason
                )

        return result

    def run_replay_backtest(
        self,
        start: date = None,
        end: date = None,
        symbols: List[str] = None,
        interval: str = "1d",
        output_dir: str = "reports/replay",
        verbose: bool = True,
        mode: ReplayMode = ReplayMode.MOCK,
        use_manifest: bool = True
    ) -> List[dict]:
        """运行回放回测

        Args:
            start: 开始日期（默认从 config 读取）
            end: 结束日期（默认从 config 读取）
            symbols: 交易标的（默认从 config 读取）
            interval: K线周期
            output_dir: 输出目录
            verbose: 是否打印详细信息
            mode: 回放模式 (MOCK/REAL)
            use_manifest: 是否使用 RunManifest 系统

        Returns:
            每日结果列表
        """
        # 从 config 获取参数（如果未提供）
        if self.config:
            start = start or datetime.strptime(self.config.start_date, "%Y-%m-%d").date()
            end = end or datetime.strptime(self.config.end_date, "%Y-%m-%d").date()
            symbols = symbols or self.config.symbols

        # 使用 RunManifest 系统
        if use_manifest and self.config:
            from run_manifest import create_run_directory, generate_experiment_id
            from utils.hash import get_git_commit, compute_config_hash, compute_data_hash

            # 构建配置字典
            config_dict = self.config.to_dict()
            config_dict['mode'] = mode.value
            git_commit = get_git_commit(short=True)

            # 创建运行目录
            run_dir, manifest = create_run_directory(
                base_dir="runs",
                config=config_dict,
                git_commit=git_commit
            )

            # 使用实验ID目录作为输出目录
            actual_output_dir = str(run_dir)
            experiment_id = manifest.experiment_id

            if verbose:
                print(f"\n实验ID: {experiment_id}")
                print(f"运行目录: {run_dir}")
        else:
            actual_output_dir = output_dir
            experiment_id = None
            manifest = None

        results = []
        market = TradingSession.get_market_type(symbols[0]) if symbols else 'US'
        engine = ReplayEngine(
            start=start, end=end, interval=interval, market=market,
            mode=mode, config=self.config
        )

        equity_curve = []
        risk_rejects = []

        if verbose:
            trading_days = TradingSession.get_trading_days(start, end, market)
            print(f"\n开始回放: {start} 到 {end}")
            print(f"回放模式: {mode.value}")
            print(f"交易标的: {', '.join(symbols)}")
            print(f"交易日数量: {len(trading_days)}")
            print(f"初始资金: ${self.broker.get_cash_balance():,.2f}\n")

        for i, point in enumerate(engine.iter_days(), 1):
            ctx = point.ctx
            daily_result = self.tick(ctx, symbols)
            results.append(daily_result)
            risk_rejects.extend(daily_result["risk_rejects"])

            closing_prices = {}
            close_dt = point.sessions[-1][1]
            for symbol in symbols:
                bars = self.data_source.get_bars_until(symbol, close_dt, interval)
                if bars:
                    closing_prices[symbol] = bars[-1].close

            if closing_prices:
                self.broker.update_position_prices(closing_prices)

            if self.db:
                for position in self.broker.get_positions().values():
                    self.db.save_position(position, ctx.trading_date)

            daily_pnl = self.broker.get_total_pnl()
            self.risk_manager.update_daily_pnl(daily_pnl)

            equity_curve.append({
                "date": ctx.trading_date,
                "cash": self.broker.get_cash_balance(),
                "equity": self.broker.get_total_equity(),
                "pnl": self.broker.get_total_pnl()
            })

            if verbose and daily_result["signals_generated"] > 0:
                print(f"[{i}] {ctx.trading_date}: "
                      f"信号={daily_result['signals_generated']}, "
                      f"提交={daily_result['orders_submitted']}, "
                      f"成交={daily_result['orders_filled']}, "
                      f"拒绝={daily_result['orders_rejected']}")

        if verbose:
            self._print_summary()

        # 写入输出文件
        self._write_replay_outputs(
            actual_output_dir, start, end, interval, symbols, equity_curve, risk_rejects,
            mode=mode, engine_metadata=engine.get_metadata() if hasattr(engine, 'get_metadata') else {}
        )

        # 更新 manifest
        if manifest:
            metrics = {
                "initial_cash": float(self.broker.initial_cash),
                "final_equity": float(self.broker.get_total_equity()),
                "total_return": float(self.broker.get_total_pnl() / self.broker.initial_cash),
                "total_pnl": float(self.broker.get_total_pnl()),
                "trade_count": len(self.broker.get_trades())
            }

            artifacts = [
                "summary.json",
                "equity_curve.csv",
                "trades.csv",
                "risk_rejects.csv"
            ]

            manifest.mark_completed(metrics=metrics, artifacts=artifacts)
            manifest.save(Path(actual_output_dir))

        return results

    def run_backtest(
        self,
        start: date,
        end: date,
        symbols: List[str],
        verbose: bool = True
    ) -> List[dict]:
        """兼容入口（默认回放）"""
        return self.run_replay_backtest(start, end, symbols, interval="1d", verbose=verbose)

    def _print_summary(self):
        """打印回测摘要"""
        print("\n" + "="*60)
        print("回测摘要")
        print("="*60)

    def _write_replay_outputs(
        self,
        output_dir: str,
        start: date,
        end: date,
        interval: str,
        symbols: List[str],
        equity_curve: List[dict],
        risk_rejects: List[dict],
        mode: ReplayMode = ReplayMode.MOCK,
        engine_metadata: Dict[str, Any] = None
    ):
        from utils.hash import get_git_commit, compute_config_hash, compute_data_hash

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 构建完整的 summary
        backtest_date = datetime.now().isoformat()

        # 数据源信息
        data_source_info = {"type": "mock"}
        data_hash = None
        if self.config:
            data_source_info = {
                "type": self.config.data_source,
                "path": self.config.data_path,
                "seed": self.config.seed if self.config.data_source == "mock" else None
            }
            # 计算数据哈希
            if self.config.data_path:
                data_hash = compute_data_hash(self.config.data_path)

        # 配置信息
        config_info = {}
        if self.config:
            config_info = {
                "data_source": self.config.data_source,
                "symbols": self.config.symbols,
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "commission": {
                    "type": self.config.commission_type,
                    "value": self.config.commission_per_share
                },
                "slippage": {
                    "enabled": self.config.slippage_enabled,
                    "type": self.config.slippage_type,
                    "value": self.config.slippage_value
                },
                "strategy": {
                    "name": self.config.strategy_name,
                    "params": self.config.strategy_params
                },
                "risk": self.config.risk_params
            }

        # 计算配置哈希
        config_hash = engine_metadata.get("config_hash") if engine_metadata else None
        if not config_hash and self.config:
            config_hash = self.config.compute_hash()

        summary = {
            "backtest_date": backtest_date,
            "replay_mode": mode.value,

            "data_source": data_source_info,
            "data_hash": data_hash,

            "config": config_info,
            "config_hash": config_hash or "unknown",
            "git_commit": engine_metadata.get("git_commit") if engine_metadata else get_git_commit(short=True),

            "date_range": {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "interval": interval
            },
            "symbols": symbols,

            "metrics": {
                "initial_cash": float(self.broker.initial_cash),
                "final_equity": float(self.broker.get_total_equity()),
                "total_return": float(self.broker.get_total_pnl() / self.broker.initial_cash),
                "total_pnl": float(self.broker.get_total_pnl()),
                "trade_count": len(self.broker.get_trades())
            }
        }

        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        equity_path = out_dir / "equity_curve.csv"
        with equity_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "cash", "equity", "pnl"])
            for row in equity_curve:
                writer.writerow([
                    row["date"].isoformat(),
                    float(row["cash"]),
                    float(row["equity"]),
                    float(row["pnl"])
                ])

        trades_path = out_dir / "trades.csv"
        with trades_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trade_id", "timestamp", "symbol", "side", "price", "quantity", "commission"])
            for trade in self.broker.get_trades():
                writer.writerow([
                    trade.trade_id,
                    trade.timestamp.isoformat(),
                    trade.symbol,
                    trade.side,
                    float(trade.price),
                    trade.quantity,
                    float(trade.commission)
                ])

        rejects_path = out_dir / "risk_rejects.csv"
        with rejects_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "symbol", "action", "reason"])
            for item in risk_rejects:
                writer.writerow([
                    item["date"].isoformat(),
                    item["symbol"],
                    item["action"],
                    item["reason"]
                ])

        self._validate_replay_outputs([
            summary_path,
            equity_path,
            trades_path,
            rejects_path
        ])

    def _validate_replay_outputs(self, paths: List[Path]):
        missing = [p for p in paths if not p.exists()]
        if missing:
            names = ", ".join(p.name for p in missing)
            raise RuntimeError(f"missing_replay_outputs: {names}")

        empty = [p for p in paths if p.stat().st_size == 0]
        if empty:
            names = ", ".join(p.name for p in empty)
            raise RuntimeError(f"empty_replay_outputs: {names}")

        print(f"\n账户状态:")
        print(f"  最终现金: ${self.broker.get_cash_balance():,.2f}")
        print(f"  总权益: ${self.broker.get_total_equity():,.2f}")
        print(f"  总盈亏: ${self.broker.get_total_pnl():,.2f} ({self.broker.get_total_pnl()/self.broker.initial_cash*100:.2f}%)")

        print(f"\n持仓情况:")
        positions = self.broker.get_positions()
        if positions:
            for symbol, pos in positions.items():
                print(f"  {symbol}: {pos.quantity}股, 成本 ${pos.avg_price}, "
                      f"市值 ${pos.market_value}, "
                      f"浮盈 ${pos.unrealized_pnl} ({pos.unrealized_pnl/pos.market_value*100:.2f}%)")
        else:
            print("  无持仓")

        print(f"\n交易统计:")
        trades = self.broker.get_trades()
        print(f"  总成交次数: {len(trades)}")
        buy_count = sum(1 for t in trades if t.side == 'BUY')
        sell_count = sum(1 for t in trades if t.side == 'SELL')
        print(f"  买入次数: {buy_count}")
        print(f"  卖出次数: {sell_count}")

        total_commission = sum(t.commission for t in trades)
        print(f"  总手续费: ${total_commission}")

        print("="*60)


def create_default_agent(
    initial_cash: float = 1000000,
    db_path: Optional[str] = None,
    log_dir: str = "logs",
    seed: int = 42,
    config: Optional[BacktestConfig] = None
) -> TradingAgent:
    """创建默认配置的Agent

    Args:
        initial_cash: 初始资金
        db_path: 数据库路径（可选）
        log_dir: 日志目录
        seed: 随机种子
        config: 回测配置（可选，会覆盖其他参数）

    Returns:
        配置好的TradingAgent
    """
    # 使用配置或默认参数
    if config:
        initial_cash = config.initial_cash
        seed = config.seed

        # 根据数据源类型创建数据源
        if config.data_source == "csv" and config.data_path:
            data_source = CSVDataSource(
                data_dir=str(Path(config.data_path).parent),
                symbol_map={s: Path(config.data_path).name for s in config.symbols}
            )
        else:
            data_source = MockMarketSource(seed=seed)

        # 创建策略
        strategy_params = config.strategy_params or {}
        if config.strategy_name == "ma":
            strategy = MAStrategy(
                fast_period=strategy_params.get("fast_period", 5),
                slow_period=strategy_params.get("slow_period", 20)
            )
        else:
            strategy = MAStrategy(fast_period=5, slow_period=20)

        # 创建风控
        risk_params = config.risk_params or {}
        risk_manager = BasicRiskManager(
            max_position_ratio=risk_params.get("max_position_ratio", 0.2),
            max_positions=risk_params.get("max_positions", 5),
            blacklist=set(risk_params.get("blacklist", [])),
            max_daily_loss_ratio=risk_params.get("max_daily_loss_ratio", 0.05)
        )

        # 创建经纪商
        broker = PaperBroker(
            initial_cash=initial_cash,
            commission_per_share=config.commission_per_share,
            slippage_bps=getattr(config, 'slippage_bps', 0.0)
        )
    else:
        # 创建默认组件
        data_source = MockMarketSource(seed=seed)
        strategy = MAStrategy(fast_period=5, slow_period=20)
        risk_manager = BasicRiskManager(
            max_position_ratio=0.2,
            max_positions=5,
            blacklist=set(),
            max_daily_loss_ratio=0.05
        )
        broker = PaperBroker(initial_cash=initial_cash, commission_per_share=0.01)

    # 创建数据库（可选）
    db = TradeDB(db_path) if db_path else None

    # 创建事件日志器
    event_logger = EventLogger(log_dir=log_dir)

    # 创建Agent
    agent = TradingAgent(data_source, strategy, risk_manager, broker, db, event_logger, config)

    return agent


def run_backtest_with_config(
    config: BacktestConfig,
    output_dir: str = "reports/replay",
    verbose: bool = True
) -> Dict[str, Any]:
    """使用配置运行回测

    Args:
        config: 回测配置
        output_dir: 输出目录
        verbose: 是否打印详细信息

    Returns:
        回测结果摘要
    """
    # 创建 Agent
    agent = create_default_agent(config=config)

    # 确定回放模式
    mode = ReplayMode.REAL if config.data_source == "csv" else ReplayMode.MOCK

    # 运行回测
    start = datetime.strptime(config.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(config.end_date, "%Y-%m-%d").date()

    results = agent.run_replay_backtest(
        start=start,
        end=end,
        symbols=config.symbols,
        interval=config.interval,
        output_dir=output_dir,
        verbose=verbose,
        mode=mode
    )

    # 读取并返回摘要
    summary_path = Path(output_dir) / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())

    return {"results_count": len(results)}


if __name__ == "__main__":
    """自测代码"""
    parser = argparse.ArgumentParser(description='运行自动交易回测')
    parser.add_argument('--start', type=str, default='2024-01-01', help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-01-31', help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--cash', type=float, default=100000, help='初始资金')
    parser.add_argument('--db', type=str, default=None, help='数据库路径')
    parser.add_argument('--interval', type=str, default='1d', help='K线周期')
    parser.add_argument('--symbols', type=str, default='AAPL,GOOGL,MSFT', help='交易标的(逗号分隔)')
    parser.add_argument('--mode', type=str, default='replay', choices=['replay', 'backtest'], help='运行模式')
    parser.add_argument('--output-dir', type=str, default='reports/replay', help='回放输出目录')
    parser.add_argument('--quiet', action='store_true', help='静默模式')

    args = parser.parse_args()

    # 解析日期
    start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    end_date = datetime.strptime(args.end, '%Y-%m-%d').date()

    # 创建Agent
    agent = create_default_agent(initial_cash=args.cash, db_path=args.db)

    # 运行回测
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]

    try:
        if args.mode == "replay":
            results = agent.run_replay_backtest(
                start=start_date,
                end=end_date,
                symbols=symbols,
                interval=args.interval,
                output_dir=args.output_dir,
                verbose=not args.quiet
            )
        else:
            results = agent.run_backtest(
                start=start_date,
                end=end_date,
                symbols=symbols,
                verbose=not args.quiet
            )

        if not args.quiet:
            print(f"\n✓ 回测完成")

    except Exception as e:
        print(f"\n✗ 回测出错: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
