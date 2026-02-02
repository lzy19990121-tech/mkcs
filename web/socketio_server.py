"""
WebSocket 服务器 - SocketIO 集成

处理实时事件推送：
- 价格更新
- 交易信号
- 成交通知
- 风控状态
"""

import logging
from datetime import datetime
from typing import Optional

from flask import Flask
from flask_socketio import SocketIO, emit, join_room, leave_room

from core.models import Signal

logger = logging.getLogger(__name__)

# SocketIO 实例
socketio: Optional[SocketIO] = None


def create_socketio(app: Flask) -> SocketIO:
    """
    创建 SocketIO 实例并注册事件处理

    Args:
        app: Flask 应用实例

    Returns:
        SocketIO 实例
    """
    global socketio

    # 配置 CORS
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",  # 开发环境允许所有跨域
        async_mode='threading',  # 使用 threading 模式配合 LiveTrader
        logger=True,
        engineio_logger=False,
    )

    # 注册事件处理
    register_handlers(socketio)

    logger.info("SocketIO 服务器已创建")
    return socketio


def register_handlers(socketio: SocketIO):
    """注册 WebSocket 事件处理"""

    @socketio.on('connect')
    def handle_connect():
        """客户端连接"""
        logger.info(f"客户端连接: {request.sid}")
        emit('connected', {'status': 'ok', 'timestamp': datetime.utcnow().isoformat()})

    @socketio.on('disconnect')
    def handle_disconnect():
        """客户端断开"""
        logger.info(f"客户端断开: {request.sid}")

    @socketio.on('subscribe')
    def handle_subscribe(data):
        """
        订阅股票数据

        Args:
            data: {'symbol': 'AAPL'} 或 {'symbols': ['AAPL', 'GOOGL']}
        """
        symbols = data.get('symbols', [])
        if isinstance(symbols, str):
            symbols = [symbols]

        if 'symbol' in data and data['symbol'] not in symbols:
            symbols.append(data['symbol'])

        for symbol in symbols:
            room = f'stock_{symbol.upper()}'
            join_room(room)
            logger.info(f"客户端 {request.sid} 订阅 {symbol}")

        emit('subscribed', {'symbols': symbols})

    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """
        取消订阅

        Args:
            data: {'symbol': 'AAPL'} 或 {'symbols': ['AAPL', 'GOOGL']}
        """
        symbols = data.get('symbols', [])
        if isinstance(symbols, str):
            symbols = [symbols]

        if 'symbol' in data and data['symbol'] not in symbols:
            symbols.append(data['symbol'])

        for symbol in symbols:
            room = f'stock_{symbol.upper()}'
            leave_room(room)
            logger.info(f"客户端 {request.sid} 取消订阅 {symbol}")

        emit('unsubscribed', {'symbols': symbols})

    @socketio.on('ping')
    def handle_ping():
        """心跳检测"""
        emit('pong', {'timestamp': datetime.utcnow().isoformat()})


def emit_price_update(symbol: str, price_data: dict):
    """
    推送价格更新

    Args:
        symbol: 股票代码
        price_data: 价格数据字典
    """
    if socketio:
        room = f'stock_{symbol.upper()}'
        socketio.emit('price_update', {
            'symbol': symbol.upper(),
            'data': price_data,
            'timestamp': datetime.utcnow().isoformat()
        }, room=room)


def emit_signal(signal: Signal):
    """
    推送交易信号

    Args:
        signal: Signal 对象
    """
    if socketio:
        socketio.emit('signal', {
            'symbol': signal.symbol,
            'action': signal.action,
            'price': float(signal.price) if signal.price else None,
            'quantity': signal.quantity,
            'confidence': signal.confidence,
            'reason': signal.reason,
            'timestamp': signal.timestamp.isoformat() if signal.timestamp else None,
        })


def emit_order(intent):
    """
    推送订单状态

    Args:
        intent: OrderIntent 对象
    """
    if socketio:
        socketio.emit('order', {
            'symbol': intent.signal.symbol if intent.signal else None,
            'action': intent.signal.action if intent.signal else None,
            'approved': intent.approved,
            'reason': intent.risk_reason,
            'timestamp': datetime.utcnow().isoformat(),
        })


def emit_fill(trade):
    """
    推送成交通知

    Args:
        trade: Trade 对象
    """
    if socketio:
        socketio.emit('trade', {
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'side': trade.side,
            'price': float(trade.price),
            'quantity': int(trade.quantity),
            'commission': float(trade.commission),
            'realized_pnl': float(trade.realized_pnl) if trade.realized_pnl else None,
            'timestamp': trade.timestamp.isoformat() if trade.timestamp else None,
        })


def emit_risk_status(status: dict):
    """
    推送风控状态

    Args:
        status: 风控状态字典
    """
    if socketio:
        socketio.emit('risk_status', {
            **status,
            'timestamp': datetime.utcnow().isoformat(),
        })


def emit_trader_status(status: dict):
    """
    推送交易器状态

    Args:
        status: 状态字典
    """
    if socketio:
        socketio.emit('trader_status', {
            **status,
            'timestamp': datetime.utcnow().isoformat(),
        })


# 延迟导入 request（需要在 Flask 上下文中）
from flask import request
