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


@orders_bp.route('/<int:order_id>', methods=['DELETE'])
def cancel_order(order_id: int):
    """撤销订单（目前仅支持取消挂单）"""
    try:
        # TODO: 实现撤销订单功能
        return jsonify({
            'status': 'error',
            'message': '撤销订单功能暂未实现'
        }), 501

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
