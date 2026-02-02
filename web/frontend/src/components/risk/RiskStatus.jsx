import React from 'react';
import { Card, Progress, Tag, Space, Statistic, Row, Col, Alert, List, Typography } from 'antd';
import {
  SafetyCertificateOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons';
import useMarketStore from '../../stores/marketStore';

const { Text } = Typography;

function RiskStatus({ riskStatus, positions = {}, performance = {} }) {
  const traderStatus = useMarketStore((state) => state.traderStatus);

  // 安全检查：确保 performance 不是 null
  const safePerformance = performance || {};

  // 计算仓位集中度
  const totalValue = Object.values(positions).reduce((sum, p) => sum + (p.market_value || 0), 0);
  const positionRatios = Object.values(positions).map((pos) => ({
    symbol: pos.symbol,
    ratio: totalValue > 0 ? (pos.market_value / totalValue) * 100 : 0,
  })).sort((a, b) => b.ratio - a.ratio);

  const maxRatio = positionRatios[0]?.ratio || 0;

  return (
    <Card
      title={
        <Space>
          <SafetyCertificateOutlined />
          风控状态
        </Space>
      }
      size="small"
    >
      {/* 交易状态 */}
      <div style={{ marginBottom: 16 }}>
        <Space>
          {riskStatus?.allowed ? (
            <Tag icon={<CheckCircleOutlined />} color="success">
              交易允许
            </Tag>
          ) : (
            <Tag icon={<CloseCircleOutlined />} color="error">
              交易禁止
            </Tag>
          )}
        </Space>
        {riskStatus?.reason && (
          <Alert
            message={riskStatus.reason}
            type={riskStatus.allowed ? 'success' : 'error'}
            showIcon
            style={{ marginTop: 8 }}
          />
        )}
      </div>

      {/* 交易器状态 */}
      {traderStatus && (
        <div style={{ marginBottom: 16 }}>
          <Text type="secondary">交易器: </Text>
          <Tag color={traderStatus.running ? 'success' : traderStatus.paused ? 'warning' : 'default'}>
            {traderStatus.running ? '运行中' : traderStatus.paused ? '已暂停' : '已停止'}
          </Tag>
        </div>
      )}

      {/* 绩效指标 */}
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Statistic
            title="总权益"
            value={safePerformance.total_equity || 0}
            prefix="$"
            precision={2}
            valueStyle={{ color: '#3fb950' }}
          />
        </Col>
        <Col span={12}>
          <Statistic
            title="今日盈亏"
            value={safePerformance.total_pnl || 0}
            prefix="$"
            precision={2}
            valueStyle={{ color: (safePerformance.total_pnl || 0) >= 0 ? '#3fb950' : '#f85149' }}
          />
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={12}>
          <Statistic
            title="交易次数"
            value={safePerformance.total_trades || 0}
          />
        </Col>
        <Col span={12}>
          <Statistic
            title="胜率"
            value={safePerformance.win_rate || 0}
            suffix="%"
            valueStyle={{ color: (safePerformance.win_rate || 0) >= 50 ? '#3fb950' : '#f85149' }}
          />
        </Col>
      </Row>

      {/* 仓位集中度 */}
      <div style={{ marginBottom: 16 }}>
        <Text type="secondary">最大单股仓位:</Text>
        <Progress
          percent={Math.min(maxRatio, 100)}
          status={maxRatio > 50 ? 'exception' : maxRatio > 30 ? 'active' : 'success'}
          size="small"
          style={{ marginTop: 4 }}
        />
        {maxRatio > 50 && (
          <Alert
            message={`${positionRatios[0]?.symbol} 仓位超过 50%`}
            type="warning"
            showIcon
            style={{ marginTop: 4 }}
          />
        )}
      </div>

      {/* 持仓明细 */}
      {positionRatios.length > 0 && (
        <div>
          <Text type="secondary" style={{ marginBottom: 8, display: 'block' }}>
            仓位分布:
          </Text>
          <List
            size="small"
            dataSource={positionRatios.slice(0, 5)}
            renderItem={(item) => (
              <List.Item>
                <Text>{item.symbol}</Text>
                <Tag color={item.ratio > 30 ? 'orange' : 'blue'}>
                  {item.ratio.toFixed(1)}%
                </Tag>
              </List.Item>
            )}
          />
        </div>
      )}

      {/* 日亏损限制 */}
      {riskStatus?.daily_loss_remaining !== undefined && (
        <div style={{ marginTop: 16 }}>
          <Text type="secondary">今日可亏损额度:</Text>
          <Progress
            percent={
              ((riskStatus.max_daily_loss - riskStatus.daily_loss_remaining) /
                riskStatus.max_daily_loss) *
              100
            }
            format={() =>
              `$${riskStatus.daily_loss_remaining?.toFixed(2)} / $${riskStatus.max_daily_loss?.toFixed(2)}`
            }
            status={
              riskStatus.daily_loss_remaining < riskStatus.max_daily_loss * 0.2
                ? 'exception'
                : 'active'
            }
            size="small"
          />
        </div>
      )}
    </Card>
  );
}

export default RiskStatus;
