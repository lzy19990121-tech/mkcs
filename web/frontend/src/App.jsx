import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import SymbolDetail from './pages/SymbolDetail';
import Sidebar from './components/Sidebar';
import Header from './components/Header';

// Lazy load new pages for better performance
const RunsList = React.lazy(() => import('./pages/RunsList'));
const RunDetail = React.lazy(() => import('./pages/RunDetail'));
const Compare = React.lazy(() => import('./pages/Compare'));
const Risk = React.lazy(() => import('./pages/Risk'));
const Rules = React.lazy(() => import('./pages/Rules'));

function App() {
  return (
    <div style={{ minHeight: '100vh', background: '#0d1117' }}>
      <Sidebar />
      <div style={{ marginLeft: 200, minHeight: '100vh' }}>
        <Header />
        <main style={{ padding: 16 }}>
          <React.Suspense fallback={<div style={{ color: '#8b949e', textAlign: 'center', padding: 50 }}>加载中...</div>}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/symbol/:symbol" element={<SymbolDetail />} />
              <Route path="/runs" element={<RunsList />} />
              <Route path="/runs/:id" element={<RunDetail />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/risk" element={<Risk />} />
              <Route path="/rules" element={<Rules />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </React.Suspense>
        </main>
      </div>
    </div>
  );
}

export default App;
