import React from 'react';
import { Empty, Button } from 'antd';
import { PlusOutlined, ReloadOutlined } from '@ant-design/icons';

/**
 * EmptyState - 空状态组件
 *
 * 统一的空状态展示组件
 */
function EmptyState({
  description = '暂无数据',
  actionText = null,
  onAction = null,
  icon = null,
}) {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 200,
        padding: 40,
        color: '#8b949e',
      }}
    >
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description={description}
        style={{ color: '#8b949e' }}
      >
        {actionText && onAction && (
          <Button type="primary" icon={icon || <PlusOutlined />} onClick={onAction}>
            {actionText}
          </Button>
        )}
      </Empty>
    </div>
  );
}

export default EmptyState;
