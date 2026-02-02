"""
风控和交易器状态 API

提供风控状态、持仓、绩效等接口
"""

from flask import Blueprint, jsonify

from web.services import live_trading_service

risk_bp = Blueprint('risk', __name__, url_prefix='/api')


@risk_bp.route('/risk/status', methods=['GET'])
def get_risk_status():
    """获取风控状态"""
    try:
        status = live_trading_service.get_risk_status()
        return jsonify(status)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@risk_bp.route('/risk/positions', methods=['GET'])
def get_positions():
    """获取所有持仓"""
    try:
        positions = live_trading_service.get_positions()
        return jsonify(positions)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@risk_bp.route('/risk/performance', methods=['GET'])
def get_performance():
    """获取绩效指标"""
    try:
        performance = live_trading_service.get_performance()
        return jsonify(performance)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@risk_bp.route('/trader/status', methods=['GET'])
def get_trader_status():
    """获取 LiveTrader 状态"""
    try:
        status = live_trading_service.get_status()
        return jsonify(status)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@risk_bp.route('/trader/control', methods=['POST'])
def control_trader():
    """
    控制 LiveTrader

    Body:
        {
            "action": "start",  // "start", "pause", "resume", "stop"
            "config": {  // 可选，启动时配置
                "symbols": ["AAPL", "GOOGL"],
                "interval": "1m"
            }
        }
    """
    from flask import request

    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        action = data.get('action')

        if action == 'start':
            live_trading_service.start()
            return jsonify({'status': 'started', 'message': 'LiveTrader 已启动'})
        elif action == 'pause':
            live_trading_service.pause()
            return jsonify({'status': 'paused', 'message': 'LiveTrader 已暂停'})
        elif action == 'resume':
            live_trading_service.resume()
            return jsonify({'status': 'resumed', 'message': 'LiveTrader 已恢复'})
        elif action == 'stop':
            live_trading_service.stop()
            return jsonify({'status': 'stopped', 'message': 'LiveTrader 已停止'})
        else:
            return jsonify({'error': f'Unknown action: {action}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@risk_bp.route('/account/summary', methods=['GET'])
def get_account_summary():
    """获取账户汇总"""
    try:
        return jsonify({
            'cash_balance': live_trading_service.get_cash_balance(),
            'total_equity': live_trading_service.get_total_equity(),
            'positions': live_trading_service.get_positions(),
            'performance': live_trading_service.get_performance(),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
