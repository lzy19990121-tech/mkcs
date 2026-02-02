import React, { useEffect } from 'react';
import { Row, Col, Card, Table, Tag, Typography, Space, Statistic, Timeline, Badge } from 'antd';
import { useNavigate } from 'react-router-dom';
import {
  DollarOutlined,
  RiseOutlined,
  FallOutlined,
  StockOutlined,
  TrophyOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import useMarketStore from '../stores/marketStore';
import { useWatchlist, usePerformance, useOrders } from '../hooks/useMarketData';
import { useWebSocket } from '../hooks/useWebSocket';

const { Text } = Typography;

function Dashboard() {
  const navigate = useNavigate();
  const watchlist = useMarketStore((state) => state.watchlist);
  const positions = useMarketStore((state) => state.positions);
  const trades = useMarketStore((state) => state.trades);
  const events = useMarketStore((state) => state.events);
  const performance = useMarketStore((state) => state.performance);
  const realtimePrices = useMarketStore((state) => state.realtimePrices);

  // 获取数据
  useWatchlist();
  usePerformance();
  useOrders();

  // WebSocket 连接
  const symbols = watchlist.map((w) => w.symbol);
  useWebSocket(symbols);

  // 合并观察列表和实时价格
  const watchlistData = watchlist.map((item) => ({
    ...item,
    price: realtimePrices[item.symbol]?.mid_price || item.quote?.mid_price,
    change: item.quote?.change,
    change_pct: item.quote?.change_pct,
    position: positions[item.symbol],
  }));

  // 表格列定义
  const columns = [
    {
      title: '股票',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (symbol, record) => (
        <Space>
          <StockOutlined />
          <Text strong>{symbol}</Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.display_name}
          </Text>
        </Space>
      ),
    },
    {
      title: '价格',
      dataIndex: 'price',
      key: 'price',
      align: 'right',
      render: (price) => `$${price?.toFixed(2) || '-'}`,
    },
    {
      title: '涨跌幅',
      dataIndex: 'change_pct',
      key: 'change_pct',
      align: 'right',
      render: (pct, record) => {
        const isPositive = (pct || record.change) >= 0;
        return (
          <Text type={isPositive ? 'success' : 'danger'}>
            {isPositive ? '+' : ''}{pct?.toFixed(2) || '0.00'}%
          </Text>
        );
      },
    },
    {
      title: '持仓',
      key: 'position',
      align: 'right',
      render: (_, record) => {
        const pos = record.position;
        if (!pos) return <Text type="secondary">-</Text>;
        return (
          <Space direction="vertical" size={0}>
            <Text>{pos.quantity} 股</Text>
            <Text
              type={pos.unrealized_pnl >= 0 ? 'success' : 'danger'}
              style={{ fontSize: 12 }}
            >
              {pos.unrealized_pnl >= 0 ? '+' : ''}${pos.unrealized_pnl?.toFixed(2)}
            </Text>
          </Space>
        );
      },
    },
    {
      title: '操作',
      key: 'action',
      align: 'center',
      render: (_, record) => (
        <Tag
          color="blue"
          style={{ cursor: 'pointer' }}
          onClick={() => navigate(`/symbol/${record.symbol}`)}
        >
          查看
        </Tag>
      ),
    },
  ];

  // 事件颜色映射
  const getEventColor = (type) => {
    switch (type) {
      case 'signal': return 'gold';
      case 'trade': return 'green';
      case 'risk': return 'red';
      case 'warning': return 'orange';
      case 'system': return 'blue';
      default: return 'gray';
    }
  };

  return (
    <div>
      <Row gutter={[16, 16]}>
        {/* 关键指标卡片 */}
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="总权益"
              value={performance?.total_equity || 100000}
              precision={2}
              prefix={<DollarOutlined />}
              valueStyle={{ color: '#3fb950' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="今日盈亏"
              value={performance?.total_pnl || 0}
              precision={2}
              prefix={(performance?.total_pnl || 0) >= 0 ? <RiseOutlined /> : <FallOutlined />}
              valueStyle={{ color: (performance?.total_pnl || 0) >= 0 ? '#3fb950' : '#f85149' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="持仓数量"
              value={Object.keys(positions).length}
              prefix={<StockOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="胜率"
              value={performance?.win_rate || 0}
              suffix="%"
              prefix={<TrophyOutlined />}
              valueStyle={{ color: (performance?.win_rate || 0) >= 50 ? '#3fb950' : '#f85149' }}
            />
          </Card>
        </Col>

        {/* 观察列表 */}
        <Col xs={24} lg={16}>
          <Card title={<><StockOutlined /> 观察列表</>}>
            <Table
              columns={columns}
              dataSource={watchlistData}
              pagination={false}
              size="small"
              rowKey="symbol"
            />
          </Card>
        </Col>

        {/* 实时事件流 */}
        <Col xs={24} lg={8}>
          <Card
            title={
              <Space>
                <ThunderboltOutlined />
                实时事件
              </Space>
            }
            extra={<Badge count={events.length} />}
          >
            <Timeline
              mode="left"
              items={events.slice(0, 10).map((event) => ({
                color: getEventColor(event.type),
                children: (
                  <div>
                    <Text style={{ fontSize: 12 }}>{event.message}</Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </Text>
                  </div>
                ),
              }))}
            />
          </Card>
        </Col>

        {/* 交易历史 */}
        <Col xs={24}>
          <Card title="最近成交">
            <Table
              columns={[
                { title: '时间', dataIndex: 'trade_time', key: 'time', render: (t) => t?.split('T')[0] + ' ' + t?.split('T')[1]?.substring(0, 8) },
                { title: '股票', dataIndex: 'symbol', key: 'symbol' },
                { title: '方向', dataIndex: 'side', key: 'side', render: (s) => <Tag color={s === 'buy' ? 'green' : 'red'}>{s.toUpperCase()}</Tag> },
                { title: '价格', dataIndex: 'price', key: 'price', render: (p) => `$${p?.toFixed(2)}` },
                { title: '数量', dataIndex: 'quantity', key: 'quantity' },
                { title: '盈亏', dataIndex: 'realized_pnl', key: 'pnl', render: (p) => <Text type={p >= 0 ? 'success' : 'danger'}>{p >= 0 ? '+' : ''}{p?.toFixed(2)}</Text> },
              ]}
              dataSource={trades.slice(0, 10)}
              pagination={false}
              size="small"
              rowKey="id"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default Dashboard;
