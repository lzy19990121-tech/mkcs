import React, { useState, useEffect, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Tabs, Card, Row, Col, Statistic, Button, Space, Typography, message, Spin, Tag, Alert } from 'antd';
import { ArrowLeftOutlined, ReloadOutlined } from '@ant-design/icons';
import { backtestsAPI } from '../services/api';
import EquityChart from '../components/runs/EquityChart';
import TradesTable from '../components/runs/TradesTable';
import RiskRejectsTable from '../components/runs/RiskRejectsTable';
import RiskRejectsChart from '../components/runs/RiskRejectsChart';
import ArtifactsBrowser from '../components/runs/ArtifactsBrowser';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

/**
 * RunDetail - 回测详情页
 *
 * 显示单个回测的完整信息，包括概览、权益曲线、交易记录、风控拒绝等
 */
function RunDetail() {
  const { id } = useParams();
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');

  // 加载回测详情
  const loadDetail = useCallback(async () => {
    if (!id) return;

    setLoading(true);
    try {
      const result = await backtestsAPI.get(id);
      setData(result);
    } catch (error) {
      if (!error.canceled) {
        message.error('加载回测详情失败: ' + (error.response?.data?.error || error.message || '未知错误'));
      }
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    loadDetail();
  }, [loadDetail]);

  // 格式化百分比
  const formatPercent = (val) => {
    if (val === undefined || val === null) return '-';
    const formatted = typeof val === 'number' ? val : parseFloat(val);
    return { value: formatted.toFixed(2), suffix: '%' };
  };

  // 渲染概览标签页
  const renderOverview = () => {
    if (!data?.summary) {
      return <div style={{ textAlign: 'center', padding: 40, color: '#8b949e' }}>暂无概览数据</div>;
    }

    const summary = data.summary;

    return (
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
          >
            <Statistic
              title={<span style={{ color: '#8b949e' }}>起始资金</span>}
              value={summary.initial_capital || summary.start_equity || 0}
              prefix="$"
              precision={2}
              valueStyle={{ color: '#c9d1d9' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
          >
            <Statistic
              title={<span style={{ color: '#8b949e' }}>最终权益</span>}
              value={summary.final_equity || 0}
              prefix="$"
              precision={2}
              valueStyle={{ color: '#c9d1d9' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
          >
            <Statistic
              title={<span style={{ color: '#8b949e' }}>总收益</span>}
              value={summary.total_return || 0}
              formatter={formatPercent}
              valueStyle={{
                color: (summary.total_return || 0) >= 0 ? '#52c41a' : '#ff4d4f',
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
          >
            <Statistic
              title={<span style={{ color: '#8b949e' }}>交易次数</span>}
              value={summary.total_trades || 0}
              valueStyle={{ color: '#c9d1d9' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
          >
            <Statistic
              title={<span style={{ color: '#8b949e' }}>胜率</span>}
              value={summary.win_rate || 0}
              formatter={(val) => `${val.toFixed(2)}%`}
              valueStyle={{ color: '#00d4aa' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
          >
            <Statistic
              title={<span style={{ color: '#8b949e' }}>夏普比率</span>}
              value={summary.sharpe_ratio || 0}
              precision={2}
              valueStyle={{ color: '#c9d1d9' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
          >
            <Statistic
              title={<span style={{ color: '#8b949e' }}>最大回撤</span>}
              value={summary.max_drawdown || 0}
              formatter={(val) => `-${val.toFixed(2)}%`}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Card
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
          >
            <Statistic
              title={<span style={{ color: '#8b949e' }}>盈利因子</span>}
              value={summary.profit_factor || 0}
              precision={2}
              valueStyle={{ color: '#c9d1d9' }}
            />
          </Card>
        </Col>

        {/* 基本信息卡片 */}
        <Col span={24}>
          <Card
            title="回测信息"
            style={{
              background: '#161b22',
              border: '1px solid #30363d',
            }}
            headStyle={{ color: '#c9d1d9', borderBottom: '1px solid #30363d' }}
          >
            <Row gutter={[16, 16]}>
              <Col xs={24} sm={8}>
                <Text type="secondary">回测 ID</Text>
                <br />
                <Text copyable style={{ color: '#00d4aa' }}>{id}</Text>
              </Col>
              <Col xs={24} sm={8}>
                <Text type="secondary">策略</Text>
                <br />
                {summary.strategy ? <Tag color="blue">{summary.strategy}</Tag> : '-'}
              </Col>
              <Col xs={24} sm={8}>
                <Text type="secondary">回测日期</Text>
                <br />
                <Text>{summary.backtest_date || summary.date || '-'}</Text>
              </Col>
              <Col xs={24} sm={8}>
                <Text type="secondary">开始日期</Text>
                <br />
                <Text>{summary.start_date || '-'}</Text>
              </Col>
              <Col xs={24} sm={8}>
                <Text type="secondary">结束日期</Text>
                <br />
                <Text>{summary.end_date || '-'}</Text>
              </Col>
              <Col xs={24} sm={8}>
                <Text type="secondary">数据周期</Text>
                <br />
                <Text>{summary.interval || '-'}</Text>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>
    );
  };

  // 渲染文件标签页
  const renderArtifacts = () => {
    return <ArtifactsBrowser backtestId={id} />;
  };

  if (loading && !data) {
    return (
      <div style={{ textAlign: 'center', padding: 100 }}>
        <Spin size="large" />
      </div>
    );
  }

  if (!data && !loading) {
    return (
      <div style={{ textAlign: 'center', padding: 100, color: '#8b949e' }}>
        未找到回测记录
      </div>
    );
  }

  const summary = data?.summary || {};

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
              回测详情: {id}
            </Title>
            {summary.strategy && (
              <Tag color="blue">{summary.strategy}</Tag>
            )}
          </Space>
          <Button
            icon={<ReloadOutlined />}
            onClick={loadDetail}
            loading={loading}
          >
            刷新
          </Button>
        </Space>
      </Card>

      {/* 内容标签页 */}
      <Card
        bordered={false}
        style={{
          background: '#161b22',
          borderRadius: 8,
        }}
        bodyStyle={{ padding: 0 }}
      >
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          style={{
            background: '#161b22',
          }}
          tabBarStyle={{
            color: '#8b949e',
            borderBottom: '1px solid #30363d',
            margin: 0,
            padding: '0 24px',
          }}
        >
          <TabPane tab="概览" key="overview">
            <div style={{ padding: 24 }}>
              {renderOverview()}
            </div>
          </TabPane>

          <TabPane tab="权益曲线" key="equity">
            <div style={{ padding: 24 }}>
              <EquityChart data={data?.equity_curve} loading={loading} />
            </div>
          </TabPane>

          <TabPane tab={`交易 (${data?.trades?.length || 0})`} key="trades">
            <div style={{ padding: 24 }}>
              <TradesTable trades={data?.trades} loading={loading} />
            </div>
          </TabPane>

          <TabPane tab={`风控拒绝 (${data?.risk_rejects?.length || 0})`} key="rejects">
            <Tabs
              defaultActiveKey="table"
              style={{ background: 'transparent' }}
              tabBarStyle={{ marginBottom: 16, paddingLeft: 0, background: 'transparent' }}
            >
              <TabPane tab="表格视图" key="table">
                <RiskRejectsTable rejects={data?.risk_rejects} loading={loading} />
              </TabPane>
              <TabPane tab="统计图表" key="chart">
                <RiskRejectsChart rejects={data?.risk_rejects} />
              </TabPane>
            </Tabs>
          </TabPane>

          <TabPane tab="文件" key="artifacts">
            <div style={{ padding: 24 }}>
              {renderArtifacts()}
            </div>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
}

export default RunDetail;
