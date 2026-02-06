import React from 'react';
import { Card, Typography, Empty } from 'antd';

const { Title } = Typography;

/**
 * Risk - 风控中心页面
 *
 * 显示风控状态和事件历史
 */
function Risk() {
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
          风控中心
        </Title>
        <Empty
          description="此功能正在开发中"
          style={{ color: '#8b949e' }}
        />
      </Card>
    </div>
  );
}

export default Risk;
