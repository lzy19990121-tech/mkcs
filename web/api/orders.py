"""
订单 API

提供订单提交、查询、撤销等接口
"""

from datetime import datetime
from flask import Blueprint, jsonify, request

from web.services import live_trading_service

orders_bp = Blueprint('orders', __name__, url_prefix='/api/orders')


@orders_bp.route('', methods=['GET'])
def list_orders():
    """获取所有订单/成交记录"""
    try:
        trades = live_trading_service.get_trades()

        # 获取当前挂单（如果有）
        positions = live_trading_service.get_positions()

        return jsonify({
            'trades': trades,
            'positions': positions,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orders_bp.route('', methods=['POST'])
def submit_order():
    """
    提交订单

    Body:
        {
            "symbol": "AAPL",
            "side": "buy",  // or "sell"
            "quantity": 10,
            "price": 150.50  // 可选，市价单可不传
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        symbol = data.get('symbol')
        side = data.get('side')
        quantity = data.get('quantity')
        price = data.get('price')

        # 参数验证
        if not symbol:
            return jsonify({'error': 'symbol is required'}), 400
        if not side or side.lower() not in ['buy', 'sell']:
            return jsonify({'error': 'side must be "buy" or "sell"'}), 400
        if not quantity or quantity <= 0:
            return jsonify({'error': 'quantity must be positive'}), 400

        result = live_trading_service.submit_order(
            symbol=symbol.upper(),
            side=side.lower(),
            quantity=quantity,
            price=price
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@orders_bp.route('/cancel', methods=['POST'])
def cancel_order_endpoint():
    """
    撤销订单

    Body:
        {
            "order_id": "123",           // 可选，订单 ID
            "client_order_id": "abc",    // 可选，客户端订单 ID
            "symbol": "AAPL",            // 可选，撤销该品种的所有订单
            "dry_run": false              // 可选，模拟运行（live 模式默认 true）
        }

    Returns:
        {
            "status": "success" | "error",
            "reason": "...",
            "cancelled_orders": [...],
            "updated_order": {...}
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        order_id = data.get('order_id')
        client_order_id = data.get('client_order_id')
        symbol = data.get('symbol')
        dry_run = data.get('dry_run', None)

        # 至少需要一个标识
        if not any([order_id, client_order_id, symbol]):
            return jsonify({
                'status': 'error',
                'reason': '至少需要提供 order_id、client_order_id 或 symbol 之一'
            }), 400

        result = live_trading_service.cancel_order(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            dry_run=dry_run
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'reason': str(e)
        }), 500


@orders_bp.route('/<int:order_id>', methods=['DELETE'])
def cancel_order_by_id(order_id: int):
    """
    撤销订单（通过 order_id）

    Query params:
        dry_run: 是否模拟运行（live 模式默认 true）
    """
    try:
        dry_run = request.args.get('dry_run', '').lower() == 'true'

        result = live_trading_service.cancel_order(
            order_id=str(order_id),
            dry_run=dry_run
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'reason': str(e)
        }), 500


@orders_bp.route('/history', methods=['GET'])
def get_trade_history():
    """
    获取交易历史

    Query params:
        symbol: 可选，筛选特定股票
        limit: 数量限制，默认 100
    """
    try:
        symbol = request.args.get('symbol')
        limit = int(request.args.get('limit', 100))

        # 从标注服务获取手动记录的交易
        from web.services import annotation_service
        trades = annotation_service.get_trades(symbol=symbol, limit=limit)

        # 合并系统成交记录
        system_trades = live_trading_service.get_trades()
        all_trades = trades + system_trades

        # 按时间排序
        all_trades.sort(key=lambda x: x.get('trade_time', ''), reverse=True)
        all_trades = all_trades[:limit]

        return jsonify(all_trades)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
