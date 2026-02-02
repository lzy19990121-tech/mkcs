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


@stocks_bp.route('/<symbol>/signals', methods=['GET'])
def get_signals(symbol: str):
    """
    获取策略信号和卖出区间

    Query params:
        interval: K线周期 - 默认 '1d'
        days: 数据天数 - 默认 90
    """
    try:
        from decimal import Decimal
        from core.models import Bar
        from skills.strategy.moving_average import MAStrategy

        interval = request.args.get('interval', '1d')
        days = int(request.args.get('days', 90))

        # 获取K线数据
        bars_data = market_data_service.get_bars_with_ma(
            symbol=symbol.upper(),
            interval=interval,
            days=days,
            ma_periods=[5, 20]
        )

        if not bars_data or not bars_data.get('bars'):
            return jsonify([])

        # 转换为 Bar 对象
        bars = []
        for bar_data in bars_data['bars']:
            bars.append(Bar(
                symbol=symbol.upper(),
                timestamp=datetime.fromisoformat(bar_data['timestamp'].replace('Z', '+00:00')),
                open=Decimal(str(bar_data['open'])),
                high=Decimal(str(bar_data['high'])),
                low=Decimal(str(bar_data['low'])),
                close=Decimal(str(bar_data['close'])),
                volume=bar_data.get('volume', 0),
                interval=interval
            ))

        # 使用策略生成信号
        strategy = MAStrategy(fast_period=5, slow_period=20)
        signals = strategy.generate_signals(bars)

        # 转换信号为 JSON
        result = []
        for signal in signals:
            # 如果有目标价格，生成卖出区间
            sell_ranges = []
            if signal.target_price:
                # 计算预期时间范围
                from datetime import timedelta
                time_horizon = signal.time_horizon or 120  # 默认120小时
                start_time = signal.timestamp
                end_time = signal.timestamp + timedelta(hours=time_horizon)

                # 创建价格区间（目标价 ± 5%）
                target_price = float(signal.target_price)
                price_range = target_price * 0.05
                upper_bound = target_price + price_range
                lower_bound = max(target_price - price_range, float(signal.price))

                sell_ranges.append({
                    'id': f"signal_{signal.symbol}_{signal.timestamp.timestamp()}",
                    'symbol': signal.symbol,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'target_price': target_price,
                    'price_range': {
                        'lower': lower_bound,
                        'upper': upper_bound
                    },
                    'stop_loss': float(signal.stop_loss) if signal.stop_loss else None,
                    'action': signal.action,
                    'reason': signal.reason,
                    'confidence': signal.confidence,
                    'current_price': float(signal.price),
                    'is_strategy_signal': True  # 标记为策略信号
                })

            result.append({
                'symbol': signal.symbol,
                'action': signal.action,
                'price': float(signal.price),
                'target_price': float(signal.target_price) if signal.target_price else None,
                'stop_loss': float(signal.stop_loss) if signal.stop_loss else None,
                'confidence': signal.confidence,
                'reason': signal.reason,
                'timestamp': signal.timestamp.isoformat(),
                'time_horizon': signal.time_horizon,
                'sell_ranges': sell_ranges
            })

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
