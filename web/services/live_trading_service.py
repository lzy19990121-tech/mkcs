"""
实时交易服务 - 封装 LiveTrader 接口

提供单例模式的 LiveTrader 管理，支持：
- 启动/暂停/停止实时交易
- 提交订单
- 获取状态、持仓、绩效
- 事件回调（信号、订单、成交）
"""

import logging
import threading
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

from agent.live_runner import LiveTrader, LiveTradingConfig, TradingMode
from broker.paper import PaperBroker
from skills.market_data.yahoo_source import YahooFinanceSource
from skills.strategy.moving_average import MAStrategy
from skills.risk.basic_risk import BasicRiskManager
from core.models import Signal, OrderIntent, Position

logger = logging.getLogger(__name__)


class LiveTradingService:
    """
    实时交易服务单例

    管理 LiveTrader 实例，提供统一的 API 接口
    """

    _instance: Optional['LiveTradingService'] = None
    _trader: Optional[LiveTrader] = None
    _callbacks: Dict[str, List[Callable]] = {
        'signal': [],
        'order': [],
        'fill': [],
        'error': [],
    }
    _trader_thread: Optional[threading.Thread] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._trader is None:
            self._initialize_trader()

    def _initialize_trader(self):
        """初始化 LiveTrader"""
        try:
            # 配置
            config = LiveTradingConfig(
                mode=TradingMode.PAPER,
                symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA'],
                interval='1m',
                check_interval_seconds=60,
            )

            # 创建组件
            data_source = YahooFinanceSource()
            strategy = MAStrategy(fast_period=5, slow_period=20)
            risk_manager = BasicRiskManager()
            broker = PaperBroker(
                initial_cash=Decimal('100000'),
                commission_per_share=Decimal('0.01'),
                slippage_bps=Decimal('10')
            )

            # 创建 LiveTrader
            self._trader = LiveTrader(
                config=config,
                data_source=data_source,
                strategy=strategy,
                risk_manager=risk_manager,
                broker=broker,
            )

            # 注册内部回调
            self._trader.register_callback('signal', self._on_signal)
            self._trader.register_callback('order', self._on_order)
            self._trader.register_callback('fill', self._on_fill)
            self._trader.register_callback('error', self._on_error)

            logger.info("LiveTrader 初始化完成")

        except Exception as e:
            logger.error(f"LiveTrader 初始化失败: {e}")
            self._trader = None

    # ============ 回调处理 ============

    def _on_signal(self, signal: Signal):
        """信号回调"""
        for cb in self._callbacks['signal']:
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    def _on_order(self, intent: OrderIntent):
        """订单回调"""
        for cb in self._callbacks['order']:
            try:
                cb(intent)
            except Exception as e:
                logger.error(f"Order callback error: {e}")

    def _on_fill(self, trade):
        """成交回调"""
        for cb in self._callbacks['fill']:
            try:
                cb(trade)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")

    def _on_error(self, error: Exception):
        """错误回调"""
        for cb in self._callbacks['error']:
            try:
                cb(error)
            except Exception as e:
                logger.error(f"Error callback error: {e}")

    # ============ 公共 API ============

    def register_callback(self, event_type: str, callback: Callable):
        """注册事件回调"""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")

    def start(self):
        """启动实时交易（在后台线程中运行）"""
        if self._trader is None:
            logger.error("LiveTrader 未初始化，无法启动")
            return

        if self._trader._running:
            logger.warning("LiveTrader 已在运行中")
            return

        if self._trader_thread and self._trader_thread.is_alive():
            logger.warning("LiveTrader 线程已在运行中")
            return

        # 在后台线程中启动
        self._trader_thread = threading.Thread(
            target=self._trader.start,
            name="LiveTrader",
            daemon=True
        )
        self._trader_thread.start()
        logger.info("LiveTrader 已在后台线程中启动")

    def pause(self):
        """暂停实时交易"""
        if self._trader:
            self._trader.pause()
            logger.info("LiveTrader 已暂停")

    def resume(self):
        """恢复实时交易"""
        if self._trader:
            self._trader.resume()
            logger.info("LiveTrader 已恢复")

    def stop(self):
        """停止实时交易"""
        if self._trader:
            self._trader.stop()
            logger.info("LiveTrader 已停止")

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._trader is not None and self._trader._running

    def get_status(self) -> Dict[str, Any]:
        """获取运行状态"""
        if self._trader is None:
            return {
                'running': False,
                'initialized': False,
                'error': 'LiveTrader 未初始化'
            }

        return {
            'running': self._trader._running,
            'initialized': True,
            'paused': self._trader._paused,
            'config': {
                'symbols': self._trader.config.symbols,
                'mode': self._trader.config.mode.value,
                'interval': self._trader.config.interval,
            }
        }

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有持仓"""
        if self._trader is None or self._trader.broker is None:
            return {}

        positions = {}
        for symbol, pos in self._trader.broker.get_positions().items():
            positions[symbol] = {
                'symbol': pos.symbol,
                'quantity': int(pos.quantity),
                'avg_price': float(pos.avg_price),
                'market_value': float(pos.market_value),
                'unrealized_pnl': float(pos.unrealized_pnl),
            }
        return positions

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取单个持仓"""
        positions = self.get_positions()
        return positions.get(symbol)

    def get_cash_balance(self) -> float:
        """获取现金余额"""
        if self._trader is None or self._trader.broker is None:
            return 0.0
        return float(self._trader.broker.get_cash_balance())

    def get_total_equity(self) -> float:
        """获取总权益"""
        if self._trader is None or self._trader.broker is None:
            return 0.0
        return float(self._trader.broker.get_total_equity())

    def get_trades(self) -> List[Dict[str, Any]]:
        """获取所有成交记录"""
        if self._trader is None or self._trader.broker is None:
            return []

        trades = []
        for trade in self._trader.broker.get_trades():
            trades.append({
                'trade_id': trade.trade_id,
                'symbol': trade.symbol,
                'side': trade.side,
                'price': float(trade.price),
                'quantity': int(trade.quantity),
                'commission': float(trade.commission),
                'realized_pnl': float(trade.realized_pnl) if trade.realized_pnl else None,
                'timestamp': trade.timestamp.isoformat() if trade.timestamp else None,
            })
        return trades

    def get_performance(self) -> Dict[str, Any]:
        """获取绩效指标"""
        if self._trader is None or self._trader.broker is None:
            return {}

        total_pnl = float(self._trader.broker.get_total_pnl())
        trades = self._trader.broker.get_trades()

        # 计算胜率
        if trades:
            winning_trades = [t for t in trades if t.realized_pnl and t.realized_pnl > 0]
            win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        else:
            win_rate = 0

        return {
            'total_equity': self.get_total_equity(),
            'cash_balance': self.get_cash_balance(),
            'total_pnl': total_pnl,
            'total_trades': len(trades),
            'win_rate': win_rate,
        }

    def submit_order(self, symbol: str, side: str, quantity: int, price: Optional[float] = None) -> Dict[str, Any]:
        """
        提交订单

        Args:
            symbol: 股票代码
            side: 'buy' or 'sell'
            quantity: 数量
            price: 可选限价（市价单可不传，会使用当前市价）

        Returns:
            Dict with status and message
        """
        if self._trader is None:
            return {'status': 'error', 'message': 'LiveTrader 未初始化'}

        try:
            # 如果没有提供价格，获取当前市场价格
            if price is None:
                try:
                    from web.services import market_data_service
                    quote = market_data_service.get_quote(symbol)
                    if quote and quote.get('mid_price'):
                        price = float(quote['mid_price'])
                    else:
                        return {'status': 'error', 'message': '无法获取当前价格，请稍后重试'}
                except Exception as e:
                    logger.error(f"获取价格失败: {e}")
                    return {'status': 'error', 'message': f'获取当前价格失败: {str(e)}'}

            # 创建信号
            signal = Signal(
                symbol=symbol,
                timestamp=datetime.utcnow(),
                action='BUY' if side.lower() == 'buy' else 'SELL',
                price=Decimal(str(price)),
                quantity=quantity,
                confidence=1.0,
                reason='Manual order'
            )

            # 风控检查
            intent = self._trader.risk_manager.check(
                signal=signal,
                positions=self._trader.broker.get_positions(),
                cash_balance=self._trader.broker.get_cash_balance()
            )

            if intent.approved:
                # 提交订单
                self._trader.broker.submit_order(intent)

                # 立即撮合订单（手动下单需要立即成交）
                from core.models import Bar
                # 创建一个虚拟的bar来触发撮合
                fill_bar = Bar(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    open=Decimal(str(price)),
                    high=Decimal(str(price)),
                    low=Decimal(str(price)),
                    close=Decimal(str(price)),
                    volume=quantity * 100,  # 假设有足够的成交量
                    interval='1m'
                )

                # 触发撮合
                fills, rejects = self._trader.broker.on_bar(fill_bar)

                # 检查是否成交
                if fills:
                    fill = fills[0]
                    return {
                        'status': 'success',
                        'message': f'订单已成交: {side.upper()} {quantity} {symbol} @ ${price:.2f}',
                        'order': {
                            'symbol': symbol,
                            'side': side,
                            'quantity': quantity,
                            'price': price,
                        },
                        'fill': {
                            'trade_id': fill.trade.trade_id,
                            'price': float(fill.trade.price),
                            'quantity': fill.trade.quantity,
                            'commission': float(fill.trade.commission)
                        }
                    }
                elif rejects:
                    return {
                        'status': 'rejected',
                        'message': f'订单被拒绝: {rejects[0].reason}',
                        'reason': rejects[0].reason
                    }
                else:
                    return {
                        'status': 'success',
                        'message': f'订单已提交: {side.upper()} {quantity} {symbol} @ ${price:.2f}',
                        'order': {
                            'symbol': symbol,
                            'side': side,
                            'quantity': quantity,
                            'price': price,
                        }
                    }
            else:
                return {
                    'status': 'rejected',
                    'message': f'订单被风控拒绝: {intent.risk_reason}',
                    'reason': intent.risk_reason,
                }

        except Exception as e:
            logger.error(f"提交订单失败: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}

    def get_risk_status(self) -> Dict[str, Any]:
        """获取风控状态"""
        if self._trader is None:
            return {
                'allowed': True,
                'reason': None,
            }

        # 检查风控管理器状态
        risk_manager = self._trader.risk_manager

        # BasicRiskManager 默认允许交易（没有持久化禁止状态）
        # 返回默认的风控状态
        return {
            'allowed': True,
            'reason': None,
        }

    def cancel_order(
        self,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        dry_run: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        撤销订单

        Args:
            order_id: 订单 ID
            client_order_id: 客户端订单 ID
            symbol: 撤销该品种的所有订单
            dry_run: 是否模拟运行（live 模式默认 true，paper 模式无效）

        Returns:
            撤销结果
        """
        if self._trader is None:
            return {
                'status': 'error',
                'reason': 'LiveTrader 未初始化'
            }

        # 检查是否是 live 模式
        is_live = self._trader.config.mode == TradingMode.LIVE
        if dry_run is None:
            dry_run = is_live  # live 模式默认 dry_run=True

        if is_live and dry_run:
            # live 模式下的模拟运行
            return {
                'status': 'success',
                'reason': 'DRY RUN: Order would be cancelled (live mode)',
                'dry_run': True,
                'cancelled_orders': [],
                'updated_order': None
            }

        try:
            # 调用 broker 的 cancel_order
            result = self._trader.broker.cancel_order(
                order_id=order_id,
                client_order_id=client_order_id,
                symbol=symbol
            )

            # 添加 dry_run 信息到结果
            if dry_run:
                result['dry_run'] = True

            return result

        except Exception as e:
            logger.error(f"撤销订单失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'reason': str(e)
            }

    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """获取所有挂单

        Returns:
            挂单列表
        """
        if self._trader is None or self._trader.broker is None:
            return []

        return self._trader.broker.get_pending_orders()


# 单例实例
live_trading_service = LiveTradingService()
