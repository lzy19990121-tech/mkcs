import React from 'react';
import { Alert, Button } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';

/**
 * ErrorState - 错误状态组件
 *
 * 统一的错误状态展示组件
 */
function ErrorState({
  message = '加载失败',
  description = null,
  onRetry = null,
}) {
  return (
    <Alert
      message={message}
      description={description}
      type="error"
      showIcon
      action={
        onRetry && (
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={onRetry}
          >
            重试
          </Button>
        )
      }
      style={{
        margin: 16,
        background: '#161b22',
        border: '1px solid #ff4d4f',
      }}
    />
  );
}

export default ErrorState;
