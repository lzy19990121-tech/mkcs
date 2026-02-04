import React, { useEffect, useState } from 'react';
import { Row, Col, Card, Table, Tag, Typography, Space, Statistic, Timeline, Badge, Tooltip } from 'antd';
import { useNavigate } from 'react-router-dom';
import {
  DollarOutlined,
  RiseOutlined,
  FallOutlined,
  StockOutlined,
  TrophyOutlined,
  ThunderboltOutlined,
  AlertOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons';
import useMarketStore from '../stores/marketStore';
import { useWatchlist, usePerformance, useOrders } from '../hooks/useMarketData';
import { useWebSocket } from '../hooks/useWebSocket';
import { stocksAPI } from '../services/api';

const { Text } = Typography;

function Dashboard() {
  const navigate = useNavigate();
  const watchlist = useMarketStore((state) => state.watchlist);
  const positions = useMarketStore((state) => state.positions);
  const trades = useMarketStore((state) => state.trades);
  const events = useMarketStore((state) => state.events);
  const performance = useMarketStore((state) => state.performance);
  const realtimePrices = useMarketStore((state) => state.realtimePrices);

  // 策略信号状态
  const [strategySignals, setStrategySignals] = useState({});

  // 辅助函数：获取所有信号
  const getAllSignals = () => {
    const signals = [];

    for (const [symbol, signalList] of Object.entries(strategySignals)) {
      if (Array.isArray(signalList)) {
        signalList.forEach(signal => {
          signals.push({
            ...signal,
            symbol: symbol,
            timestamp: signal.timestamp || new Date().toISOString()
          });
        });
      } else if (signalList && typeof signalList === 'object') {
        signals.push({
          ...signalList,
          symbol: symbol,
          timestamp: signalList.timestamp || new Date().toISOString()
        });
      }
    }

    signals.sort((a, b) => {
      const timeA = new Date(a.timestamp).getTime();
      const timeB = new Date(b.timestamp).getTime();
      return timeB - timeA;
    });

    return signals.slice(0,20);
  };

  // 获取数据
  useWatchlist();
  usePerformance();
  useOrders();

  // WebSocket 连接
  const symbols = watchlist.map((w) => w.symbol);
  useWebSocket(symbols);

  // 获取策略信号
  useEffect(() => {
    const fetchSignals = async () => {
      const signals = {};
      for (const symbol of symbols) {
        try {
          const data = await stocksAPI.getSignals(symbol, { interval: '1d', days: 90 });
          if (data && data.length > 0) {
            signals[symbol] = data[0]; // 取最新信号
          }
        } catch (error) {
          if (!error.canceled) {
            console.error(`Failed to fetch signals for ${symbol}:`, error);
          }
        }
      }
      setStrategySignals(signals);
    };

    if (symbols.length > 0) {
      fetchSignals();
      // 每30秒刷新一次信号
      const interval = setInterval(fetchSignals, 30000);
      return () => clearInterval(interval);
    }
  }, [symbols]);

  // 检查价格是否触及目标价或止损价
  const getPriceAlert = (symbol, currentPrice) => {
    const signal = strategySignals[symbol];
    if (!signal || !signal.sell_ranges || signal.sell_ranges.length === 0) {
      return null;
    }

    const range = signal.sell_ranges[0];
    const targetPrice = range.target_price;
    const stopLoss = range.stop_loss;

    if (!targetPrice || !stopLoss || !currentPrice) return null;

    // 计算距离目标价和止损价的百分比
    const distanceToTarget = ((targetPrice - currentPrice) / currentPrice) * 100;
    const distanceToStop = ((currentPrice - stopLoss) / currentPrice) * 100;

    // 如果价格在目标价的 ±2% 范围内
    if (Math.abs(distanceToTarget) <= 2) {
      return {
        type: 'target',
        message: `当前: $${currentPrice.toFixed(2)} | 目标: $${targetPrice.toFixed(2)} | 差距: ${distanceToTarget.toFixed(1)}%`,
        shortMessage: `接近目标 $${targetPrice.toFixed(2)}`,
        color: 'green',
        currentPrice,
        targetPrice,
        stopLoss,
        distanceToTarget,
        distanceToStop
      };
    }

    // 如果价格触及或跌破止损价
    if (distanceToStop <= 0) {
      return {
        type: 'stop',
        message: `当前: $${currentPrice.toFixed(2)} | 止损: $${stopLoss.toFixed(2)} | 已跌破 ${Math.abs(distanceToStop).toFixed(1)}%`,
        shortMessage: `触及止损 $${stopLoss.toFixed(2)}`,
        color: 'red',
        currentPrice,
        targetPrice,
        stopLoss,
        distanceToTarget,
        distanceToStop
      };
    }

    // 如果距离止损价很近（2%以内）
    if (distanceToStop <= 2) {
      return {
        type: 'warning',
        message: `当前: $${currentPrice.toFixed(2)} | 止损: $${stopLoss.toFixed(2)} | 差距: ${distanceToStop.toFixed(1)}%`,
        shortMessage: `接近止损 $${stopLoss.toFixed(2)}`,
        color: 'orange',
        currentPrice,
        targetPrice,
        stopLoss,
        distanceToTarget,
        distanceToStop
      };
    }

    return null;
  };

  // 合并观察列表和实时价格
  const watchlistData = watchlist.map((item) => {
    const currentPrice = realtimePrices[item.symbol]?.price || realtimePrices[item.symbol]?.mid_price || item.price;
    const alert = getPriceAlert(item.symbol, currentPrice);

    return {
      ...item,
      price: currentPrice,
      change: item.change,
      change_pct: item.change_pct,
      position: positions[item.symbol],
      alert,
      signal: strategySignals[item.symbol],
    };
  });

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
      title: '目标/止损',
      key: 'target_stop',
      align: 'center',
      width: 200,
      render: (_, record) => {
        const signal = record.signal;
        if (!signal || !signal.sell_ranges || signal.sell_ranges.length === 0) {
          return <Text type="secondary" style={{ fontSize: 11 }}>-</Text>;
        }

        const range = signal.sell_ranges[0];
        const targetPrice = range.target_price;
        const stopLoss = range.stop_loss;
        const currentPrice = record.price;

        // 计算进度条（当前价格相对于目标价和止损价的位置）
        const totalRange = targetPrice - stopLoss;
        const currentPosition = currentPrice - stopLoss;
        const progressPercent = Math.max(0, Math.min(100, (currentPosition / totalRange) * 100));

        return (
          <Tooltip
            title={
              <div style={{ fontSize: 12 }}>
                <div style={{ marginBottom: 4, paddingBottom: 4, borderBottom: '1px solid rgba(255,255,255,0.2)' }}>
                  <strong>提醒时刻价格: ${record.alert?.currentPrice?.toFixed(2) || currentPrice?.toFixed(2)}</strong>
                </div>
                <div>当前价: <strong>${currentPrice?.toFixed(2)}</strong></div>
                <div>目标价: <span style={{ color: '#52c41a' }}>${targetPrice?.toFixed(2)}</span></div>
                <div>止损价: <span style={{ color: '#f5222d' }}>${stopLoss?.toFixed(2)}</span></div>
                <div>距离目标: {(((targetPrice - currentPrice) / currentPrice) * 100).toFixed(1)}%</div>
                <div>距离止损: {(((currentPrice - stopLoss) / currentPrice) * 100).toFixed(1)}%</div>
              </div>
            }
          >
            <div style={{ cursor: 'pointer' }}>
              <div style={{ fontSize: 11, marginBottom: 2 }}>
                <Text type="success" strong>${targetPrice?.toFixed(1)}</Text>
                <Text type="secondary" style={{ margin: '0 4px' }}>/</Text>
                <Text type="danger" strong>${stopLoss?.toFixed(1)}</Text>
              </div>
              <div
                style={{
                  width: '100%',
                  height: 4,
                  backgroundColor: '#f0f0f0',
                  borderRadius: 2,
                  position: 'relative',
                }}
              >
                <div
                  style={{
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    height: '100%',
                    width: `${progressPercent}%`,
                    backgroundColor: progressPercent > 80 ? '#52c41a' : progressPercent < 20 ? '#f5222d' : '#faad14',
                    borderRadius: 2,
                    transition: 'all 0.3s',
                  }}
                />
              </div>
            </div>
          </Tooltip>
        );
      },
    },
    {
      title: '提醒',
      key: 'alert',
      align: 'center',
      width: 100,
      render: (_, record) => {
        if (record.alert) {
          return (
            <Tooltip title={record.alert.message}>
              <Tag
                color={record.alert.color}
                icon={record.alert.type === 'target' ? <CheckCircleOutlined /> : <AlertOutlined />}
                style={{ cursor: 'pointer' }}
              >
                {record.alert.shortMessage}
              </Tag>
            </Tooltip>
          );
        }

        // 显示策略信号状态
        if (record.signal) {
          const signal = record.signal;
          return (
            <Tooltip title={`${signal.reason} (置信度: ${(signal.confidence * 100).toFixed(0)}%)`}>
              <Tag color="blue" style={{ fontSize: 11 }}>
                {signal.action === 'BUY' ? '买入信号' : '卖出信号'}
              </Tag>
            </Tooltip>
          );
        }

        return <Text type="secondary" style={{ fontSize: 11 }}>-</Text>;
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
        <Col xs={24} lg={12}>
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

        {/* 信号总值_注册_run_inv_point_checksum_recorder */}
        <Col xs={24} lg={12}>
          <Card title={
            <Space>
              <span>📊 信号记录</span>
              <Tag color="blue">历史</Tag>
            </Space>
          }>
            <Table
              columns={[
                { title: '时间', dataIndex: 'timestamp', key: 'timestamp', width: 130, render: (t) => { if (!t) return '-'; const d = new Date(t); return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }); } },
                { title: '股票', dataIndex: 'symbol', key: 'symbol', width: 80, render: (s) => <Text strong>{s}</Text> },
                { title: '方向', dataIndex: 'action', key: 'action', width: 70, render: (a) => <Tag color={a === 'BUY' ? 'green' : 'red'} style={{ fontWeight: 'bold' }}>{a === 'BUY' ? '买入' : '卖出'}</Tag> },
                { title: '价格', dataIndex: 'price', key: 'price', width: 80, align: 'right', render: (p) => `$${p?.toFixed(2)}` },
                { title: '目标价', dataIndex: 'target_price', key: 'target_price', width: 80, align: 'right', render: (p) => p ? `$${p.toFixed(2)}` : '-' },
                { title: '止损', dataIndex: 'stop_loss', key: 'stop_loss', width: 80, align: 'right', render: (p) => p ? `$${p.toFixed(2)}` : '-' },
                { title: '置信度', dataIndex: 'confidence', key: 'confidence', width: 70, align: 'center', render: (c) => <Tag color={c >= 0.7 ? 'green' : c >= 0.5 ? 'orange' : 'blue'}>{(c * 100).toFixed(0)}%</Tag> },
                { title: '原因', dataIndex: 'reason', key: 'reason', ellipsis: true, render: (r) => <Tooltip title={r}><Text style={{ fontSize: 11 }}>{r}</Text></Tooltip> },
              ]}
              dataSource={getAllSignals()}
              pagination={false}
              size="small"
              rowKey={(record) => `${record.symbol}-${record.timestamp}`}
              scroll={{ y: 240 }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default Dashboard;
