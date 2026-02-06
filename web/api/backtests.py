"""
回测 API - 提供回测结果、文件清单、报告读取等功能
"""

import json
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from flask import Blueprint, jsonify, request, send_from_directory

logger = logging.getLogger(__name__)

backtests_bp = Blueprint('backtests', __name__, url_prefix='/api/backtest')


def _get_backtest_dir(backtest_id: str) -> Optional[Path]:
    """
    获取回测目录路径

    Args:
        backtest_id: 回测ID

    Returns:
        Path对象，如果不存在则返回None
    """
    from flask import current_app
    data_dir = Path(current_app.config.get('DATA_DIR', 'data'))
    backtest_dir = data_dir / backtest_id

    if backtest_dir.exists():
        return backtest_dir
    return None


def _load_summary_from_dir(backtest_dir: Path) -> Dict:
    """从目录加载 summary.json"""
    summary_file = backtest_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _load_equity_curve(file_path: Path) -> List[Dict]:
    """加载权益曲线数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过表头
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 2:
            try:
                data.append({
                    'timestamp': parts[0],
                    'equity': float(parts[1]),
                    'cash': float(parts[2]) if len(parts) > 2 else None
                })
            except ValueError:
                continue

    return data


def _load_trades(file_path: Path) -> List[Dict]:
    """加载交易记录"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过表头
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 6:
            try:
                pnl = float(parts[6]) if len(parts) > 6 and parts[6] else None
                data.append({
                    'timestamp': parts[0],
                    'symbol': parts[1],
                    'side': parts[2],
                    'price': float(parts[3]),
                    'quantity': int(parts[4]),
                    'commission': float(parts[5]),
                    'pnl': pnl
                })
            except (ValueError, IndexError):
                continue

    return data


def _load_risk_rejects(file_path: Path) -> List[Dict]:
    """加载风控拒绝记录"""
    data = []
    if not file_path.exists():
        return data

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过表头
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            try:
                confidence = float(parts[4]) if len(parts) > 4 and parts[4] else None
                data.append({
                    'timestamp': parts[0],
                    'symbol': parts[1],
                    'action': parts[2],
                    'reason': parts[3],
                    'confidence': confidence
                })
            except (ValueError, IndexError):
                continue

    return data


def _list_artifacts_recursive(directory: Path, base_path: Path) -> List[Dict]:
    """递归列出目录中的文件"""
    artifacts = []

    for item in directory.iterdir():
        relative_path = item.relative_to(base_path)

        if item.is_dir():
            artifacts.append({
                'name': item.name,
                'type': 'directory',
                'path': str(relative_path),
                'children': _list_artifacts_recursive(item, base_path)
            })
        elif item.is_file():
            artifacts.append({
                'name': item.name,
                'type': 'file',
                'path': str(relative_path),
                'size': item.stat().st_size,
                'extension': item.suffix.lower()
            })

    # 按名称排序，目录在前
    artifacts.sort(key=lambda x: (x['type'] == 'file', x['name'].lower()))

    return artifacts


@backtests_bp.route('/<backtest_id>/artifacts', methods=['GET'])
def list_artifacts(backtest_id: str):
    """
    返回回测目录的文件清单

    Query params:
        flat: 是否返回扁平化列表 (默认 false)
    """
    backtest_dir = _get_backtest_dir(backtest_id)

    if not backtest_dir:
        return jsonify({'error': 'Backtest not found'}), 404

    flat = request.args.get('flat', 'false').lower() == 'true'

    try:
        if flat:
            # 扁平化列表
            artifacts = []
            for item in backtest_dir.rglob('*'):
                if item.is_file():
                    artifacts.append({
                        'name': item.name,
                        'type': 'file',
                        'path': str(item.relative_to(backtest_dir)),
                        'size': item.stat().st_size,
                        'extension': item.suffix.lower()
                    })
            artifacts.sort(key=lambda x: x['path'].lower())
            return jsonify(artifacts)
        else:
            # 树形结构
            artifacts = _list_artifacts_recursive(backtest_dir, backtest_dir)
            return jsonify(artifacts)

    except Exception as e:
        logger.error(f"列出文件失败: {e}")
        return jsonify({'error': str(e)}), 500


@backtests_bp.route('/<backtest_id>/artifact/<path:filepath>', methods=['GET'])
def get_artifact(backtest_id: str, filepath: str):
    """
    读取文件内容

    Query params:
        download: 是否作为下载返回 (默认 false)
    """
    backtest_dir = _get_backtest_dir(backtest_id)

    if not backtest_dir:
        return jsonify({'error': 'Backtest not found'}), 404

    # 安全检查：确保 filepath 在 backtest_dir 内
    file_path = (backtest_dir / filepath).resolve()
    if not str(file_path).startswith(str(backtest_dir.resolve())):
        return jsonify({'error': 'Invalid file path'}), 400

    if not file_path.exists():
        return jsonify({'error': 'File not found'}), 404

    if not file_path.is_file():
        return jsonify({'error': 'Not a file'}), 400

    try:
        # 根据文件扩展名决定如何处理
        ext = file_path.suffix.lower()
        download = request.args.get('download', 'false').lower() == 'true'

        if download:
            # 直接下载文件
            return send_from_directory(
                backtest_dir,
                filepath,
                as_attachment=True,
                mimetype='application/octet-stream'
            )

        if ext in ['.json']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            return jsonify({
                'type': 'json',
                'content': content
            })

        elif ext in ['.md', '.txt', '.log']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({
                'type': 'text',
                'content': content,
                'encoding': 'utf-8'
            })

        elif ext == '.csv':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({
                'type': 'csv',
                'content': content,
                'encoding': 'utf-8'
            })

        elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
            return send_from_directory(
                backtest_dir,
                filepath,
                mimetype=f'image/{ext[1:]}'
            )

        else:
            # 其他文件类型，返回基本信息
            return jsonify({
                'type': 'binary',
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'message': 'File type not supported for preview, use download=true to download'
            })

    except Exception as e:
        logger.error(f"读取文件失败: {e}")
        return jsonify({'error': str(e)}), 500


@backtests_bp.route('/<backtest_id>/report', methods=['GET'])
def get_report(backtest_id: str):
    """
    获取回测报告（自动寻找优先级最高的报告文件）

    优先级: risk_report.md > deep_risk_report.md > report.md > summary.json
    """
    backtest_dir = _get_backtest_dir(backtest_id)

    if not backtest_dir:
        return jsonify({'error': 'Backtest not found'}), 404

    # 按优先级查找报告文件
    report_files = [
        'risk_report.md',
        'deep_risk_report.md',
        'report.md',
        'summary.json'
    ]

    for report_file in report_files:
        file_path = backtest_dir / report_file
        if file_path.exists():
            try:
                if file_path.suffix == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    return jsonify({
                        'type': 'json',
                        'file': report_file,
                        'content': content
                    })
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return jsonify({
                        'type': 'markdown',
                        'file': report_file,
                        'content': content
                    })
            except Exception as e:
                logger.error(f"读取报告文件 {report_file} 失败: {e}")
                continue

    return jsonify({'error': 'No report found'}), 404


__all__ = [
    'backtests_bp',
    'list_artifacts',
    'get_artifact',
    'get_report',
]
