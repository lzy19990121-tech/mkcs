import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import SymbolDetail from './pages/SymbolDetail';
import Sidebar from './components/Sidebar';
import Header from './components/Header';

function App() {
  return (
    <div style={{ minHeight: '100vh', background: '#0d1117' }}>
      <Sidebar />
      <div style={{ marginLeft: 200, minHeight: '100vh' }}>
        <Header />
        <main style={{ padding: 16 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/symbol/:symbol" element={<SymbolDetail />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default App;
