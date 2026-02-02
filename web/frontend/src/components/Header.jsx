import React from 'react';
import { Layout, Button, Space, Tag } from 'antd';
import { useNavigate } from 'react-router-dom';
import { PlayCircleOutlined, PauseCircleOutlined, ReloadOutlined } from '@ant-design/icons';
import useMarketStore from '../stores/marketStore';
import { riskAPI } from '../services/api';

const { Header: AntHeader } = Layout;

function Header() {
  const navigate = useNavigate();
  const riskStatus = useMarketStore((state) => state.riskStatus);
  const traderStatus = useMarketStore((state) => state.traderStatus);
  const events = useMarketStore((state) => state.events);

  const handleStart = async () => {
    try {
      await riskAPI.controlTrader('start');
    } catch (error) {
      console.error('Failed to start trader:', error);
    }
  };

  const handlePause = async () => {
    try {
      await riskAPI.controlTrader('pause');
    } catch (error) {
      console.error('Failed to pause trader:', error);
    }
  };

  const handleResume = async () => {
    try {
      await riskAPI.controlTrader('resume');
    } catch (error) {
      console.error('Failed to resume trader:', error);
    }
  };

  const latestEvent = events[0];

  return (
    <AntHeader
      style={{
        padding: '0 24px',
        background: '#0d1117',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid #30363d',
        height: 64,
      }}
    >
      <div>
        <Space size="middle">
          {traderStatus?.running ? (
            <>
              <Button
                icon={<PauseCircleOutlined />}
                onClick={handlePause}
                style={{ borderColor: '#f85149', color: '#f85149' }}
              >
                暂停
              </Button>
              <Tag color="success">运行中</Tag>
            </>
          ) : traderStatus?.paused ? (
            <>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={handleResume}
              >
                恢复
              </Button>
              <Tag color="warning">已暂停</Tag>
            </>
          ) : (
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={handleStart}
            >
              启动交易
            </Button>
          )}

          <Button
            icon={<ReloadOutlined />}
            onClick={() => window.location.reload()}
          >
            刷新
          </Button>
        </Space>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        {/* 风控状态 */}
        {riskStatus && (
          <Tag color={riskStatus.allowed ? 'success' : 'error'}>
            {riskStatus.allowed ? '✓ 交易允许' : `✗ ${riskStatus.reason || '风控限制'}`}
          </Tag>
        )}

        {/* 最新事件 */}
        {latestEvent && (
          <span style={{ color: '#8b949e', fontSize: 12 }}>
            {latestEvent.message}
          </span>
        )}
      </div>
    </AntHeader>
  );
}

export default Header;
