import React, { useState, useEffect, useCallback } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { Card, Row, Col, Statistic, Button, Space, Typography, message, Spin, Tag, Alert } from 'antd';
import { ArrowLeftOutlined, ReloadOutlined } from '@ant-design/icons';
import { backtestsAPI } from '../services/api';
import EquityOverlayChart from '../components/compare/EquityOverlayChart';

const { Title, Text } = Typography;

/**
 * Compare - 策略对比页面
 *
 * 对比两个回测运行结果的性能指标
 */
function Compare() {
  const [searchParams] = useSearchParams();
  const leftId = searchParams.get('l');
  const rightId = searchParams.get('r');

  const [loading, setLoading] = useState(false);
  const [leftData, setLeftData] = useState(null);
  const [rightData, setRightData] = useState(null);

  // 加载对比数据
  const loadCompareData = useCallback(async () => {
    if (!leftId || !rightId) {
      message.warning('请选择两条回测记录进行对比');
      return;
    }

    setLoading(true);
    try {
      const [left, right] = await Promise.all([
        backtestsAPI.get(leftId),
        backtestsAPI.get(rightId),
      ]);
      setLeftData(left);
      setRightData(right);
    } catch (error) {
      if (!error.canceled) {
        message.error('加载对比数据失败');
      }
    } finally {
      setLoading(false);
    }
  }, [leftId, rightId]);

  useEffect(() => {
    loadCompareData();
  }, [loadCompareData]);

  // 格式化差异值
  const renderDiff = (leftVal, rightVal, format = 'percent') => {
    if (leftVal === undefined || rightVal === undefined) return null;

    const left = typeof leftVal === 'number' ? leftVal : parseFloat(leftVal || 0);
    const right = typeof rightVal === 'number' ? rightVal : parseFloat(rightVal || 0);
    const diff = right - left;
    const diffPercent = left !== 0 ? (diff / Math.abs(left)) * 100 : 0;

    const color = diff > 0 ? '#52c41a' : diff < 0 ? '#ff4d4f' : '#8b949e';

    return (
      <div style={{ marginTop: 4 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          差异: {diff > 0 ? '+' : ''}{format === 'percent' ? diff.toFixed(2) + '%' : diff.toFixed(2)}
        </Text>
        <Text style={{ fontSize: 12, color, marginLeft: 8 }}>
          ({diffPercent > 0 ? '+' : ''}{diffPercent.toFixed(1)}%)
        </Text>
      </div>
    );
  };

  // 对比卡片
  const ComparisonCard = ({ title, leftValue, rightValue, format = 'number', suffix = '' }) => {
    const formatValue = (val) => {
      if (val === undefined || val === null) return '-';
      if (format === 'percent') return `${typeof val === 'number' ? val.toFixed(2) : val}%`;
      if (format === 'money') return `$${typeof val === 'number' ? val.toLocaleString('en-US', { minimumFractionDigits: 2 }) : val}`;
      return typeof val === 'number' ? val.toFixed(2) : val;
    };

    return (
      <Card
        title={title}
        size="small"
        style={{
          background: '#161b22',
          border: '1px solid #30363d',
        }}
        headStyle={{ borderBottom: '1px solid #30363d', color: '#8b949e', fontSize: 12 }}
      >
        <Row gutter={16}>
          <Col span={11}>
            <div style={{ textAlign: 'center' }}>
              <Text type="secondary" style={{ fontSize: 12 }}>左侧</Text>
              <div style={{ fontSize: 20, fontWeight: 'bold', color: '#00d4aa' }}>
                {formatValue(leftValue)}{suffix}
              </div>
            </div>
          </Col>
          <Col span={2} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Text type="secondary" style={{ fontSize: 16 }}>vs</Text>
          </Col>
          <Col span={11}>
            <div style={{ textAlign: 'center' }}>
              <Text type="secondary" style={{ fontSize: 12 }}>右侧</Text>
              <div style={{ fontSize: 20, fontWeight: 'bold', color: '#ff9800' }}>
                {formatValue(rightValue)}{suffix}
              </div>
            </div>
          </Col>
        </Row>
        {renderDiff(leftValue, rightValue, format)}
      </Card>
    );
  };

  if (!leftId || !rightId) {
    return (
      <div style={{ padding: '0 0 24px' }}>
        <Card
          bordered={false}
          style={{
            background: '#161b22',
            borderRadius: 8,
            minHeight: 400,
          }}
        >
          <Alert
            message="请选择两条回测记录进行对比"
            description="从回测列表中选择两条记录，然后点击「对比选中」按钮。"
            type="info"
            showIcon
            action={
              <Link to="/runs">
                <Button type="primary">前往回测列表</Button>
              </Link>
            }
          />
        </Card>
      </div>
    );
  }

  if (loading && !leftData && !rightData) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    );
  }

  const leftSummary = leftData?.summary || {};
  const rightSummary = rightData?.summary || {};

  return (
    <div style={{ padding: '0 0 24px' }}>
      {/* 页头 */}
      <Card
        bordered={false}
        style={{
          background: '#161b22',
          borderRadius: 8,
          marginBottom: 16,
        }}
        bodyStyle={{ padding: '16px 24px' }}
      >
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Space>
            <Link to="/runs">
              <Button icon={<ArrowLeftOutlined />}>返回列表</Button>
            </Link>
            <Title level={4} style={{ margin: 0, color: '#c9d1d9' }}>
              回测对比
            </Title>
          </Space>
          <Button
            icon={<ReloadOutlined />}
            onClick={loadCompareData}
            loading={loading}
          >
            刷新
          </Button>
        </Space>
      </Card>

      {/* 对比信息 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Card
            size="small"
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
              borderLeft: '4px solid #00d4aa',
            }}
          >
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              <Text type="secondary">左侧回测</Text>
              <Text style={{ color: '#00d4aa', fontWeight: 'bold' }}>{leftId}</Text>
              {leftSummary.strategy && <Tag color="blue">{leftSummary.strategy}</Tag>}
              <Text type="secondary" style={{ fontSize: 12 }}>
                {leftSummary.start_date || '-'} 至 {leftSummary.end_date || '-'}
              </Text>
            </Space>
          </Card>
        </Col>
        <Col span={12}>
          <Card
            size="small"
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
              borderLeft: '4px solid #ff9800',
            }}
          >
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              <Text type="secondary">右侧回测</Text>
              <Text style={{ color: '#ff9800', fontWeight: 'bold' }}>{rightId}</Text>
              {rightSummary.strategy && <Tag color="blue">{rightSummary.strategy}</Tag>}
              <Text type="secondary" style={{ fontSize: 12 }}>
                {rightSummary.start_date || '-'} 至 {rightSummary.end_date || '-'}
              </Text>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 关键指标对比 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} md={8}>
          <ComparisonCard
            title="总收益率"
            leftValue={leftSummary.total_return}
            rightValue={rightSummary.total_return}
            format="percent"
          />
        </Col>
        <Col xs={24} sm={12} md={8}>
          <ComparisonCard
            title="夏普比率"
            leftValue={leftSummary.sharpe_ratio}
            rightValue={rightSummary.sharpe_ratio}
            format="number"
          />
        </Col>
        <Col xs={24} sm={12} md={8}>
          <ComparisonCard
            title="最大回撤"
            leftValue={leftSummary.max_drawdown}
            rightValue={rightSummary.max_drawdown}
            format="percent"
          />
        </Col>
        <Col xs={24} sm={12} md={8}>
          <ComparisonCard
            title="最终权益"
            leftValue={leftSummary.final_equity}
            rightValue={rightSummary.final_equity}
            format="money"
          />
        </Col>
        <Col xs={24} sm={12} md={8}>
          <ComparisonCard
            title="交易次数"
            leftValue={leftSummary.total_trades}
            rightValue={rightSummary.total_trades}
            format="number"
          />
        </Col>
        <Col xs={24} sm={12} md={8}>
          <ComparisonCard
            title="胜率"
            leftValue={leftSummary.win_rate}
            rightValue={rightSummary.win_rate}
            format="percent"
          />
        </Col>
      </Row>

      {/* 权益曲线对比 */}
      <Card
        title="权益曲线对比"
        style={{
          background: '#161b22',
          border: '1px solid #30363d',
        }}
        headStyle={{ borderBottom: '1px solid #30363d', color: '#c9d1d9' }}
      >
        <EquityOverlayChart
          leftData={leftData?.equity_curve}
          rightData={rightData?.equity_curve}
          leftLabel={leftId}
          rightLabel={rightId}
          height={400}
        />
      </Card>

      {/* 交易统计对比 */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card
            title="交易统计"
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
            headStyle={{ borderBottom: '1px solid #30363d', color: '#c9d1d9' }}
          >
            <Row gutter={16}>
              <Col xs={12} sm={6}>
                <div style={{ textAlign: 'center', padding: 16 }}>
                  <Text type="secondary">左侧盈利交易</Text>
                  <div style={{ fontSize: 24, color: '#00d4aa', fontWeight: 'bold' }}>
                    {leftSummary.winning_trades || '-'}
                  </div>
                </div>
              </Col>
              <Col xs={12} sm={6}>
                <div style={{ textAlign: 'center', padding: 16 }}>
                  <Text type="secondary">右侧盈利交易</Text>
                  <div style={{ fontSize: 24, color: '#ff9800', fontWeight: 'bold' }}>
                    {rightSummary.winning_trades || '-'}
                  </div>
                </div>
              </Col>
              <Col xs={12} sm={6}>
                <div style={{ textAlign: 'center', padding: 16 }}>
                  <Text type="secondary">左侧亏损交易</Text>
                  <div style={{ fontSize: 24, color: '#00d4aa', fontWeight: 'bold' }}>
                    {leftSummary.losing_trades || '-'}
                  </div>
                </div>
              </Col>
              <Col xs={12} sm={6}>
                <div style={{ textAlign: 'center', padding: 16 }}>
                  <Text type="secondary">右侧亏损交易</Text>
                  <div style={{ fontSize: 24, color: '#ff9800', fontWeight: 'bold' }}>
                    {rightSummary.losing_trades || '-'}
                  </div>
                </div>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default Compare;
