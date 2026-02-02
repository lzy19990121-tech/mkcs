import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Row, Col, Card, Button, Space, Modal, Form, Input, InputNumber, Select, message, Typography, List, Tag, DatePicker } from 'antd';
import { ArrowLeftOutlined, PlusOutlined, DeleteOutlined } from '@ant-design/icons';
import CandlestickChart from '../components/charts/CandlestickChart';
import OrderPanel from '../components/trading/OrderPanel';
import PositionCard from '../components/trading/PositionCard';
import RiskStatus from '../components/risk/RiskStatus';
import useMarketStore from '../stores/marketStore';
import { useWatchlist, useBars, useOrders, usePerformance } from '../hooks/useMarketData';
import { useWebSocket } from '../hooks/useWebSocket';
import { stocksAPI, annotationsAPI } from '../services/api';

const { Text, Title } = Typography;

function SymbolDetail() {
  const { symbol } = useParams();
  const navigate = useNavigate();
  const [interval, setIntervalValue] = useState('1d');

  // 状态管理
  const positions = useMarketStore((state) => state.positions);
  const performance = useMarketStore((state) => state.performance);
  const riskStatus = useMarketStore((state) => state.riskStatus);
  const realtimePrices = useMarketStore((state) => state.realtimePrices);
  const addEvent = useMarketStore((state) => state.addEvent);

  // 数据获取
  useWatchlist();
  useOrders();
  usePerformance();

  // 获取 K 线数据
  const { bars, maData, loading, refetch: refetchBars } = useBars(symbol, interval, 90);

  // 获取卖出区间
  useEffect(() => {
    const fetchSellRanges = async () => {
      try {
        console.log('Fetching sell ranges for:', symbol);
        const data = await annotationsAPI.getRanges(symbol);
        console.log('Sell ranges data:', data);
        setSellRanges(data);
      } catch (error) {
        console.error('Failed to fetch sell ranges:', error);
      }
    };
    fetchSellRanges();
  }, [symbol]);

  // WebSocket 订阅
  useWebSocket([symbol]);

  // 当前价格
  const currentPrice = realtimePrices[symbol]?.price || realtimePrices[symbol]?.mid_price || bars?.[bars.length - 1]?.close;
  const position = positions[symbol];

  // 添加标注弹窗
  const [markerModalVisible, setMarkerModalVisible] = useState(false);
  const [markerForm] = Form.useForm();
  const [pendingMarker, setPendingMarker] = useState(null);

  // 卖出区间状态
  const [sellRanges, setSellRanges] = useState([]);
  const [strategySignals, setStrategySignals] = useState([]);
  const [rangeModalVisible, setRangeModalVisible] = useState(false);
  const [rangeForm] = Form.useForm();

  // 获取策略信号和卖出区间
  useEffect(() => {
    const fetchSignals = async () => {
      try {
        console.log('Fetching strategy signals for:', symbol);
        const signals = await stocksAPI.getSignals(symbol, { interval, days: 90 });
        console.log('Strategy signals:', signals);

        setStrategySignals(signals);

        // 提取策略生成的卖出区间
        const strategySellRanges = [];
        signals.forEach(signal => {
          if (signal.sell_ranges && signal.sell_ranges.length > 0) {
            strategySellRanges.push(...signal.sell_ranges);
          }
        });

        // 合并手动添加的卖出区间和策略生成的卖出区间
        setSellRanges(prevRanges => {
          // 移除之前的策略信号，保留手动添加的
          const manualRanges = prevRanges.filter(r => !r.is_strategy_signal);
          return [...manualRanges, ...strategySellRanges];
        });
      } catch (error) {
        console.error('Failed to fetch strategy signals:', error);
      }
    };
    fetchSignals();
  }, [symbol, interval]);

  // 获取手动添加的卖出区间
  useEffect(() => {
    const fetchManualRanges = async () => {
      try {
        const data = await annotationsAPI.getRanges(symbol);
        // 只添加手动标注的区间
        setSellRanges(prevRanges => {
          const strategyRanges = prevRanges.filter(r => r.is_strategy_signal);
          return [...strategyRanges, ...data];
        });
      } catch (error) {
        console.error('Failed to fetch manual sell ranges:', error);
      }
    };
    fetchManualRanges();
  }, [symbol]);

  // 处理添加标注
  const handleAddMarker = (data) => {
    setPendingMarker(data);
    setMarkerModalVisible(true);
  };

  // 处理添加卖出区间
  const handleAddSellRange = () => {
    setRangeModalVisible(true);
  };

  const handleSubmitSellRange = async (values) => {
    try {
      await annotationsAPI.addRange(symbol, {
        start_time: values.start_time.format(),
        end_time: values.end_time.format(),
        target_price: values.target_price,
        notes: values.notes,
      });
      message.success('卖出区间已添加');

      // 刷新卖出区间列表
      const data = await annotationsAPI.getRanges(symbol);
      setSellRanges(data);

      // 刷新图表
      refetchBars();
      setRangeModalVisible(false);
      rangeForm.resetFields();
    } catch (error) {
      message.error('添加卖出区间失败: ' + (error.response?.data?.error || error.message));
    }
  };

  // 处理删除卖出区间
  const handleDeleteRange = async (rangeId) => {
    try {
      await annotationsAPI.deleteRange(symbol, rangeId);
      message.success('卖出区间已删除');

      // 刷新列表
      const data = await annotationsAPI.getRanges(symbol);
      setSellRanges(data);

      // 刷新图表
      refetchBars();
    } catch (error) {
      message.error('删除失败: ' + (error.response?.data?.error || error.message));
    }
  };

  const handleSubmitMarker = async (values) => {
    try {
      await stocksAPI.addMarker(symbol, {
        marker_type: values.type,
        price: pendingMarker.price,
        timestamp: pendingMarker.time,
        quantity: values.quantity,
        notes: values.notes,
      });
      message.success('标注已添加');
      refetchBars();
      setMarkerModalVisible(false);
      markerForm.resetFields();
    } catch (error) {
      message.error('添加标注失败');
    }
  };

  // 处理交易提交
  const handleOrderSubmit = (result) => {
    addEvent({
      type: result.status === 'success' ? 'trade' : 'warning',
      message: result.message,
    });
    useOrders(); // 刷新数据
  };

  return (
    <div>
      {/* 顶部导航 */}
      <div style={{ marginBottom: 16, display: 'flex', alignItems: 'center' }}>
        <Button
          icon={<ArrowLeftOutlined />}
          onClick={() => navigate('/')}
        >
          返回仪表盘
        </Button>
        <Title level={4} style={{ margin: '0 0 0 16px' }}>
          {symbol}
          <Text style={{ marginLeft: 16, fontSize: 14 }}>
            ${currentPrice?.toFixed(2) || '-'}
          </Text>
        </Title>
      </div>

      <Row gutter={[16, 16]}>
        {/* K 线图 */}
        <Col xs={24}>
          <Card>
            <CandlestickChart
              symbol={symbol}
              bars={bars}
              maData={maData}
              markers={[]}
              sellRanges={sellRanges}
              loading={loading}
              interval={interval}
              onAddMarker={handleAddMarker}
              onIntervalChange={setIntervalValue}
            />
          </Card>
        </Col>

        {/* 卖出区间 */}
        <Col xs={24}>
          <Card
            title={
              <Space>
                卖出区间
                <Button size="small" icon={<PlusOutlined />} onClick={handleAddSellRange}>
                  添加区间
                </Button>
              </Space>
            }
            extra={
              <Tag color={sellRanges.length > 0 ? 'processing' : 'default'}>
                {sellRanges.length} 个区间
              </Tag>
            }
          >
            {sellRanges.length === 0 ? (
              <Text type="secondary">暂无卖出区间</Text>
            ) : (
              <List
                size="small"
                dataSource={sellRanges}
                renderItem={(range) => (
                  <List.Item
                    style={{
                      backgroundColor: range.is_strategy_signal ? '#f6ffed' : 'transparent',
                      padding: '12px',
                      borderRadius: '4px'
                    }}
                    actions={[
                      !range.is_strategy_signal && (
                        <Button
                          size="small"
                          danger
                          icon={<DeleteOutlined />}
                          onClick={() => handleDeleteRange(range.id)}
                        >
                          删除
                        </Button>
                      )
                    ]}
                  >
                    <Space direction="vertical" size={4} style={{ width: '100%' }}>
                      {range.is_strategy_signal && (
                        <Tag color="green">策略信号</Tag>
                      )}
                      <div>
                        <Text strong>
                          {range.is_strategy_signal ? '策略建议' : '手动标注'}
                        </Text>
                        {range.action && (
                          <Tag color={range.action === 'BUY' ? 'green' : 'red'} style={{ marginLeft: 8 }}>
                            {range.action === 'BUY' ? '买入' : '卖出'}
                          </Tag>
                        )}
                      </div>
                      {range.price_range ? (
                        <div>
                          <Text type="secondary">
                            目标区间: ${range.price_range.lower?.toFixed(2)} - ${range.price_range.upper?.toFixed(2)}
                          </Text>
                        </div>
                      ) : (
                        <div>
                          <Text type="secondary">
                            {range.start_time?.split('T')[0]} 至 {range.end_time?.split('T')[0]}
                          </Text>
                        </div>
                      )}
                      <div>
                        <Text>
                          目标价: <Text strong style={{ color: '#52c41a' }}>${range.target_price?.toFixed(2) || 'N/A'}</Text>
                        </Text>
                        {range.stop_loss && (
                          <Text style={{ marginLeft: 16 }}>
                            止损: <Text type="danger">${range.stop_loss.toFixed(2)}</Text>
                          </Text>
                        )}
                      </div>
                      {range.confidence && (
                        <div>
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            置信度: {(range.confidence * 100).toFixed(0)}%
                          </Text>
                        </div>
                      )}
                      {range.reason && (
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {range.reason}
                        </Text>
                      )}
                      {range.notes && !range.is_strategy_signal && (
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          备注: {range.notes}
                        </Text>
                      )}
                    </Space>
                  </List.Item>
                )}
              />
            )}
          </Card>
        </Col>

        {/* 交易面板 */}
        <Col xs={24} lg={8}>
          <OrderPanel
            symbol={symbol}
            currentPrice={currentPrice}
            position={position}
            onOrderSubmit={handleOrderSubmit}
          />
        </Col>

        {/* 持仓和风控 */}
        <Col xs={24} lg={16}>
          <Row gutter={[16, 16]}>
            <Col xs={24}>
              <PositionCard
                positions={positions}
                onClosePosition={(sym, side, qty) => {
                  stocksAPI.submit({ symbol: sym, side, quantity: qty }).then((res) => {
                    message.info(res.message);
                    handleOrderSubmit(res);
                  });
                }}
              />
            </Col>
            <Col xs={24}>
              <RiskStatus
                riskStatus={riskStatus}
                positions={positions}
                performance={performance}
              />
            </Col>
          </Row>
        </Col>
      </Row>

      {/* 添加标注弹窗 */}
      <Modal
        title={<Space><PlusOutlined /> 添加标注</Space>}
        open={markerModalVisible}
        onCancel={() => {
          setMarkerModalVisible(false);
          markerForm.resetFields();
        }}
        onOk={() => markerForm.submit()}
      >
        <Form form={markerForm} layout="vertical" onFinish={handleSubmitMarker}>
          <Form.Item name="type" label="类型" rules={[{ required: true }]}>
            <Select
              options={[
                { value: 'buy', label: '买入点' },
                { value: 'sell', label: '卖出点' },
                { value: 'entry', label: '入场信号' },
                { value: 'exit', label: '出场信号' },
              ]}
            />
          </Form.Item>
          <Form.Item label="价格">
            <InputNumber
              style={{ width: '100%' }}
              value={pendingMarker?.price}
              disabled
              prefix="$"
            />
          </Form.Item>
          <Form.Item name="quantity" label="数量（可选）">
            <InputNumber style={{ width: '100%' }} min={1} />
          </Form.Item>
          <Form.Item name="notes" label="备注">
            <Input.TextArea rows={3} placeholder="添加说明..." />
          </Form.Item>
        </Form>
      </Modal>

      {/* 添加卖出区间弹�� */}
      <Modal
        title={<Space><PlusOutlined /> 添加卖出区间</Space>}
        open={rangeModalVisible}
        onCancel={() => {
          setRangeModalVisible(false);
          rangeForm.resetFields();
        }}
        onOk={() => rangeForm.submit()}
        width={500}
      >
        <Form form={rangeForm} layout="vertical" onFinish={handleSubmitSellRange}>
          <Form.Item
            name="start_time"
            label="开始时间"
            rules={[{ required: true, message: '请选择开始时间' }]}
          >
            <DatePicker
              style={{ width: '100%' }}
              showTime
              format="YYYY-MM-DD HH:mm"
              placeholder="选择开始时间"
            />
          </Form.Item>
          <Form.Item
            name="end_time"
            label="结束时间"
            rules={[{ required: true, message: '请选择结束时间' }]}
          >
            <DatePicker
              style={{ width: '100%' }}
              showTime
              format="YYYY-MM-DD HH:mm"
              placeholder="选择结束时间"
            />
          </Form.Item>
          <Form.Item
            name="target_price"
            label="目标价格"
            rules={[{ required: true, message: '请输入目标价格' }]}
          >
            <InputNumber
              style={{ width: '100%' }}
              min={0}
              step={0.01}
              precision={2}
              prefix="$"
              placeholder="例如: 150.00"
            />
          </Form.Item>
          <Form.Item name="notes" label="备注">
            <Input.TextArea rows={3} placeholder="添加说明..." />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
}

export default SymbolDetail;
