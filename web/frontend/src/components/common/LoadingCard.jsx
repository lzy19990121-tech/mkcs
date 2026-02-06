import React from 'react';
import { Card, Spin } from 'antd';

/**
 * LoadingCard - 加载状态卡片
 *
 * 统一的加载状态展示组件
 */
function LoadingCard({ message = '加载中...' }) {
  return (
    <Card
      style={{
        background: '#161b22',
        border: '1px solid #30363d',
        borderRadius: 8,
        minHeight: 200,
      }}
      bodyStyle={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 200,
      }}
    >
      <Spin size="large" tip={message} />
    </Card>
  );
}

export default LoadingCard;
