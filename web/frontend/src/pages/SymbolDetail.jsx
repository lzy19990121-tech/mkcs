import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Row, Col, Card, Button, Space, Modal, Form, InputNumber, Select, message, Typography } from 'antd';
import { ArrowLeftOutlined, PlusOutlined } from '@ant-design/icons';
import CandlestickChart from '../components/charts/CandlestickChart';
import OrderPanel from '../components/trading/OrderPanel';
import PositionCard from '../components/trading/PositionCard';
import RiskStatus from '../components/risk/RiskStatus';
import useMarketStore from '../stores/marketStore';
import { useWatchlist, useBars, useOrders, usePerformance } from '../hooks/useMarketData';
import { useWebSocket } from '../hooks/useWebSocket';
import { stocksAPI } from '../services/api';

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

  // WebSocket 订阅
  useWebSocket([symbol]);

  // 当前价格
  const currentPrice = realtimePrices[symbol]?.mid_price || bars?.[bars.length - 1]?.close;
  const position = positions[symbol];

  // 添加标注弹窗
  const [markerModalVisible, setMarkerModalVisible] = useState(false);
  const [markerForm] = Form.useForm();
  const [pendingMarker, setPendingMarker] = useState(null);

  // 处理添加标注
  const handleAddMarker = (data) => {
    setPendingMarker(data);
    setMarkerModalVisible(true);
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
              loading={loading}
              interval={interval}
              onAddMarker={handleAddMarker}
              onIntervalChange={setIntervalValue}
            />
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
    </div>
  );
}

export default SymbolDetail;
