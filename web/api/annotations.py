"""
标注 API

提供买卖点和卖出区间的 CRUD 操作
"""

from datetime import datetime
from flask import Blueprint, jsonify, request

from web.services import annotation_service

annotations_bp = Blueprint('annotations', __name__, url_prefix='/api/annotations')


# ============ 买卖点标注 ============

@annotations_bp.route('/<symbol>/markers', methods=['GET'])
def get_markers(symbol: str):
    """获取指定股票的买卖点标注"""
    try:
        markers = annotation_service.get_markers(symbol.upper())
        return jsonify(markers)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@annotations_bp.route('/<symbol>/markers', methods=['POST'])
def add_marker(symbol: str):
    """
    添加买卖点标注

    Body:
        {
            "marker_type": "buy",  // "buy", "sell", "entry", "exit"
            "price": 150.50,
            "timestamp": "2024-01-15T10:30:00",  // ISO 格式
            "quantity": 10,  // 可选
            "notes": "买入理由"  // 可选
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        marker_type = data.get('marker_type')
        price = data.get('price')
        timestamp_str = data.get('timestamp')
        quantity = data.get('quantity')
        notes = data.get('notes')

        # 参数验证
        if not marker_type or marker_type not in ['buy', 'sell', 'entry', 'exit']:
            return jsonify({'error': 'marker_type must be "buy", "sell", "entry", or "exit"'}), 400
        if price is None:
            return jsonify({'error': 'price is required'}), 400
        if not timestamp_str:
            return jsonify({'error': 'timestamp is required'}), 400

        # 解析时间戳
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({'error': 'Invalid timestamp format'}), 400

        marker = annotation_service.add_marker(
            symbol=symbol.upper(),
            marker_type=marker_type,
            price=float(price),
            timestamp=timestamp,
            quantity=int(quantity) if quantity else None,
            notes=notes
        )

        return jsonify(marker), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@annotations_bp.route('/<symbol>/markers/<int:marker_id>', methods=['PUT'])
def update_marker(symbol: str, marker_id: int):
    """
    更新标注

    Body (所有字段可选):
        {
            "price": 155.00,
            "quantity": 20,
            "notes": "更新说明"
        }
    """
    try:
        data = request.get_json() or {}

        marker = annotation_service.update_marker(
            marker_id=marker_id,
            price=data.get('price'),
            quantity=data.get('quantity'),
            notes=data.get('notes')
        )

        if marker is None:
            return jsonify({'error': 'Marker not found'}), 404

        return jsonify(marker)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@annotations_bp.route('/<symbol>/markers/<int:marker_id>', methods=['DELETE'])
def delete_marker(symbol: str, marker_id: int):
    """删除标注"""
    try:
        success = annotation_service.delete_marker(marker_id)

        if not success:
            return jsonify({'error': 'Marker not found'}), 404

        return jsonify({'status': 'deleted'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============ 卖出区间 ============

@annotations_bp.route('/<symbol>/ranges', methods=['GET'])
def get_ranges(symbol: str):
    """获取指定股票的卖出区间"""
    try:
        ranges = annotation_service.get_sell_ranges(symbol.upper())
        return jsonify(ranges)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@annotations_bp.route('/<symbol>/ranges', methods=['POST'])
def add_range(symbol: str):
    """
    添加卖出区间

    Body:
        {
            "start_time": "2024-01-15T10:30:00",  // ISO 格式
            "end_time": "2024-01-15T14:00:00",    // ISO 格式
            "target_price": 160.00,  // 可选
            "notes": "目标价位"       // 可选
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        start_time_str = data.get('start_time')
        end_time_str = data.get('end_time')
        target_price = data.get('target_price')
        notes = data.get('notes')

        # 参数验证
        if not start_time_str:
            return jsonify({'error': 'start_time is required'}), 400
        if not end_time_str:
            return jsonify({'error': 'end_time is required'}), 400

        # 解析时间戳
        try:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        except ValueError:
            return jsonify({'error': 'Invalid timestamp format'}), 400

        sell_range = annotation_service.add_sell_range(
            symbol=symbol.upper(),
            start_time=start_time,
            end_time=end_time,
            target_price=float(target_price) if target_price else None,
            notes=notes
        )

        return jsonify(sell_range), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@annotations_bp.route('/<symbol>/ranges/<int:range_id>', methods=['PUT'])
def update_range(symbol: str, range_id: int):
    """
    更新卖出区间

    Body (所有字段可选):
        {
            "start_time": "2024-01-15T11:00:00",
            "end_time": "2024-01-15T15:00:00",
            "target_price": 165.00,
            "notes": "更新说明"
        }
    """
    try:
        data = request.get_json() or {}

        # 解析时间戳
        start_time = None
        end_time = None

        if data.get('start_time'):
            try:
                start_time = datetime.fromisoformat(data['start_time'].replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'error': 'Invalid start_time format'}), 400

        if data.get('end_time'):
            try:
                end_time = datetime.fromisoformat(data['end_time'].replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'error': 'Invalid end_time format'}), 400

        sell_range = annotation_service.update_sell_range(
            range_id=range_id,
            start_time=start_time,
            end_time=end_time,
            target_price=data.get('target_price'),
            notes=data.get('notes')
        )

        if sell_range is None:
            return jsonify({'error': 'Range not found'}), 404

        return jsonify(sell_range)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@annotations_bp.route('/<symbol>/ranges/<int:range_id>', methods=['DELETE'])
def delete_range(symbol: str, range_id: int):
    """删除卖出区间"""
    try:
        success = annotation_service.delete_sell_range(range_id)

        if not success:
            return jsonify({'error': 'Range not found'}), 404

        return jsonify({'status': 'deleted'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
