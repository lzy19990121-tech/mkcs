import React from 'react';
import { Card, Typography, Empty } from 'antd';

const { Title } = Typography;

/**
 * Rules - 规则管理页面
 *
 * 用于管理和应用不同版本的交易规则
 */
function Rules() {
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
        <Title level={4} style={{ color: '#c9d1d9' }}>
          规则管理
        </Title>
        <Empty
          description="此功能正在开发中"
          style={{ color: '#8b949e' }}
        />
      </Card>
    </div>
  );
}

export default Rules;
