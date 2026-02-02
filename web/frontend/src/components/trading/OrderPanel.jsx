import React, { useState } from 'react';
import { Card, InputNumber, Input, Button, Space, Alert, Typography, Row, Col, Select } from 'antd';
import { DollarOutlined, StockOutlined } from '@ant-design/icons';
import { ordersAPI } from '../../services/api';

const { Text } = Typography;

function OrderPanel({ symbol, currentPrice, position, onOrderSubmit }) {
  const [side, setSide] = useState('buy');
  const [quantity, setQuantity] = useState(1);
  const [price, setPrice] = useState(null);
  const [notes, setNotes] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    if (!quantity || quantity <= 0) {
      setResult({ type: 'error', message: '请输入有效的数量' });
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      const response = await ordersAPI.submit({
        symbol,
        side,
        quantity,
        price: price || undefined,
        notes: notes || undefined,
      });

      setResult({
        type: response.status === 'success' ? 'success' : 'warning',
        message: response.message,
      });

      if (response.status === 'success') {
        onOrderSubmit?.(response);
        // 清空表单
        setNotes('');
      }
    } catch (error) {
      setResult({
        type: 'error',
        message: error.response?.data?.error || '订单提交失败',
      });
    } finally {
      setLoading(false);
    }
  };

  // 估算金额
  const estimatedAmount = (price || currentPrice || 0) * quantity;

  // 检查是否可以卖出
  const canSell = position && position.quantity >= quantity;

  return (
    <Card
      title={
        <Space>
          <StockOutlined />
          交易面板 - {symbol}
        </Space>
      }
      size="small"
    >
      {result && (
        <Alert
          message={result.message}
          type={result.type}
          showIcon
          style={{ marginBottom: 16 }}
          closable
          onClose={() => setResult(null)}
        />
      )}

      {/* 买卖方向 */}
      <div style={{ marginBottom: 16 }}>
        <Text type="secondary" style={{ marginRight: 8 }}>方向:</Text>
        <Space>
          <Button
            type={side === 'buy' ? 'primary' : 'default'}
            onClick={() => setSide('buy')}
            style={{ background: side === 'buy' ? '#3fb950' : undefined }}
          >
            买入
          </Button>
          <Button
            type={side === 'sell' ? 'primary' : 'default'}
            onClick={() => setSide('sell')}
            danger
            disabled={!canSell && side === 'sell'}
          >
            卖出
          </Button>
        </Space>
      </div>

      {/* 当前持仓提示 */}
      {side === 'sell' && position && (
        <Alert
          message={`当前持仓: ${position.quantity} 股，成本价 $${position.avg_price?.toFixed(2)}`}
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
      )}

      <Row gutter={16}>
        <Col span={12}>
          <div style={{ marginBottom: 8 }}>
            <Text type="secondary">
              <DollarOutlined /> 价格
            </Text>
          </div>
          <InputNumber
            size="large"
            style={{ width: '100%' }}
            min={0}
            step={0.01}
            precision={2}
            placeholder="市价"
            value={price}
            onChange={setPrice}
            prefix="$"
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {currentPrice ? `市价: $${currentPrice.toFixed(2)}` : '输入限价'}
          </Text>
        </Col>
        <Col span={12}>
          <div style={{ marginBottom: 8 }}>
            <Text type="secondary">
              <StockOutlined /> 数量
            </Text>
          </div>
          <InputNumber
            size="large"
            style={{ width: '100%' }}
            min={1}
            step={1}
            value={quantity}
            onChange={setQuantity}
          />
        </Col>
      </Row>

      {/* 估算金额 */}
      <div style={{ margin: '16px 0', textAlign: 'right' }}>
        <Text strong>
          估算: ${estimatedAmount.toFixed(2)}
        </Text>
      </div>

      {/* 备注 */}
      <Input
        placeholder="交易备注（可选）"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        style={{ marginBottom: 16 }}
      />

      {/* 提交按钮 */}
      <Button
        type="primary"
        size="large"
        block
        loading={loading}
        onClick={handleSubmit}
        style={{
          background: side === 'buy' ? '#3fb950' : '#f85149',
          borderColor: side === 'buy' ? '#3fb950' : '#f85149',
        }}
      >
        {side === 'buy' ? '买入' : '卖出'} {symbol} × {quantity}
      </Button>
    </Card>
  );
}

export default OrderPanel;
