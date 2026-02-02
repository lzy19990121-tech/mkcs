import React from 'react';
import { Card, Table, Tag, Space, Button, Typography } from 'antd';
import { DollarOutlined } from '@ant-design/icons';

const { Text } = Typography;

function PositionCard({ positions = {}, onClosePosition }) {
  const columns = [
    {
      title: '股票',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (symbol) => <Text strong>{symbol}</Text>,
    },
    {
      title: '持仓',
      dataIndex: 'quantity',
      key: 'quantity',
      align: 'right',
    },
    {
      title: '成本价',
      dataIndex: 'avg_price',
      key: 'avg_price',
      align: 'right',
      render: (price) => `$${price?.toFixed(2) || '0.00'}`,
    },
    {
      title: '当前价',
      dataIndex: 'current_price',
      key: 'current_price',
      align: 'right',
      render: (price, record) => {
        const current = price || record.market_value / record.quantity;
        return `$${current?.toFixed(2) || '0.00'}`;
      },
    },
    {
      title: '市值',
      dataIndex: 'market_value',
      key: 'market_value',
      align: 'right',
      render: (value) => `$${value?.toFixed(2) || '0.00'}`,
    },
    {
      title: '浮动盈亏',
      dataIndex: 'unrealized_pnl',
      key: 'unrealized_pnl',
      align: 'right',
      render: (pnl) => {
        const isPositive = pnl >= 0;
        return (
          <Text type={isPositive ? 'success' : 'danger'}>
            {isPositive ? '+' : ''}{pnl?.toFixed(2) || '0.00'}
          </Text>
        );
      },
    },
    {
      title: '操作',
      key: 'action',
      align: 'right',
      render: (_, record) => (
        <Space>
          <Button
            size="small"
            onClick={() => onClosePosition?.(record.symbol, 'sell', record.quantity)}
          >
            平仓
          </Button>
        </Space>
      ),
    },
  ];

  const data = Object.values(positions).map((pos) => ({
    ...pos,
    key: pos.symbol,
  }));

  if (data.length === 0) {
    return (
      <Card title={<><DollarOutlined /> 持仓</>}>
        <Text type="secondary">暂无持仓</Text>
      </Card>
    );
  }

  // 计算汇总
  const totalValue = data.reduce((sum, p) => sum + (p.market_value || 0), 0);
  const totalPnl = data.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0);

  return (
    <Card
      title={
        <Space>
          <DollarOutlined />
          <span>持仓 ({data.length})</span>
          <Tag color={totalPnl >= 0 ? 'success' : 'error'}>
            总额: ${totalValue.toFixed(2)} | 浮盈: {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(2)}
          </Tag>
        </Space>
      }
      size="small"
    >
      <Table
        columns={columns}
        dataSource={data}
        pagination={false}
        size="small"
        scroll={{ x: 800 }}
      />
    </Card>
  );
}

export default PositionCard;
