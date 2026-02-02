"""
股票数据 API

提供 K 线数据、实时报价、观察列表等接口
"""

from datetime import datetime
from flask import Blueprint, jsonify, request

from web.services import market_data_service, annotation_service

stocks_bp = Blueprint('stocks', __name__, url_prefix='/api/stocks')


@stocks_bp.route('', methods=['GET'])
def list_stocks():
    """获取观察列表"""
    try:
        watchlist = annotation_service.get_watchlist()

        # 获取实时价格
        symbols = [item['symbol'] for item in watchlist]
        quotes = market_data_service.get_batch_quotes(symbols)

        # 合并数据
        result = []
        for item in watchlist:
            symbol = item['symbol']
            quote = quotes.get(symbol, {})

            # 计算涨跌幅（如果有昨收）
            change = None
            change_pct = None
            if quote.get('prev_close'):
                if quote.get('mid_price'):
                    change = quote['mid_price'] - quote['prev_close']
                    change_pct = (change / quote['prev_close']) * 100 if quote['prev_close'] else None

            result.append({
                'symbol': symbol,
                'display_name': item['display_name'],
                'price': quote.get('mid_price'),
                'bid': quote.get('bid_price'),
                'ask': quote.get('ask_price'),
                'change': change,
                'change_pct': change_pct,
                'volume': quote.get('volume'),
                'timestamp': quote.get('timestamp'),
            })

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stocks_bp.route('/<symbol>', methods=['GET'])
def get_stock(symbol: str):
    """获取单个股票信息"""
    try:
        quote = market_data_service.get_quote(symbol.upper())
        position = None

        # 获取持仓信息（如果交易服务已启动）
        try:
            from web.services import live_trading_service
            position = live_trading_service.get_position(symbol.upper())
        except Exception:
            pass

        return jsonify({
            'symbol': symbol.upper(),
            'quote': quote,
            'position': position,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stocks_bp.route('/<symbol>/bars', methods=['GET'])
def get_bars(symbol: str):
    """
    获取 K 线数据

    Query params:
        interval: K 线周期 ('1m', '5m', '15m', '1h', '1d', '1wk', '1mo') - 默认 '1d'
        days: 天数 - 默认 90
        ma: MA 周期，逗号分隔（如 '5,20'）- 默认 '5,20'
    """
    try:
        interval = request.args.get('interval', '1d')
        days = int(request.args.get('days', 90))
        ma_periods = [int(p) for p in request.args.get('ma', '5,20').split(',') if p]

        result = market_data_service.get_bars_with_ma(
            symbol=symbol.upper(),
            interval=interval,
            days=days,
            ma_periods=ma_periods
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stocks_bp.route('/<symbol>/quote', methods=['GET'])
def get_quote(symbol: str):
    """获取实时报价"""
    try:
        quote = market_data_service.get_quote(symbol.upper())
        if quote:
            return jsonify(quote)
        return jsonify({'error': 'Quote not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stocks_bp.route('/<symbol>/price', methods=['GET'])
def get_price(symbol: str):
    """获取最新价格"""
    try:
        price = market_data_service.get_latest_price(symbol.upper())
        if price:
            return jsonify({'symbol': symbol.upper(), 'price': price})
        return jsonify({'error': 'Price not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
