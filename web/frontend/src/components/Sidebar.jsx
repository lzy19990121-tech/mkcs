import React from 'react';
import { Layout, Menu } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  HomeOutlined,
  LineChartOutlined,
  ExperimentOutlined,
  DiffOutlined,
  SafetyOutlined,
  SettingOutlined,
} from '@ant-design/icons';

const { Sider } = Layout;

function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <HomeOutlined />,
      label: '仪表盘',
    },
    {
      key: '/dashboard',
      icon: <LineChartOutlined />,
      label: '交易面板',
    },
    {
      key: '/runs',
      icon: <ExperimentOutlined />,
      label: '回测研究',
    },
    {
      key: '/compare',
      icon: <DiffOutlined />,
      label: '策略对比',
    },
    {
      key: '/risk',
      icon: <SafetyOutlined />,
      label: '风控中心',
    },
    {
      key: '/rules',
      icon: <SettingOutlined />,
      label: '规则管理',
    },
  ];

  return (
    <Sider
      collapsible
      collapsedWidth={80}
      style={{
        overflow: 'auto',
        height: '100vh',
        position: 'fixed',
        left: 0,
        top: 0,
        bottom: 0,
        background: '#161b22',
        borderRight: '1px solid #30363d',
      }}
    >
      <div
        style={{
          height: 64,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderBottom: '1px solid #30363d',
        }}
      >
        <span
          style={{
            color: '#00d4aa',
            fontSize: 20,
            fontWeight: 'bold',
            letterSpacing: 1,
          }}
        >
          MKCS
        </span>
      </div>
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={({ key }) => navigate(key)}
        style={{ background: 'transparent', borderRight: 'none' }}
      />
    </Sider>
  );
}

export default Sidebar;
