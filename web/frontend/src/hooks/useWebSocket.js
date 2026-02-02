import { useEffect, useRef, useCallback } from 'react';
import { io } from 'socket.io-client';
import useMarketStore from '../stores/marketStore';

const SOCKET_URL = window.location.hostname === 'localhost'
  ? 'http://localhost:5000'
  : window.location.origin;

let socket = null;

export function getSocket() {
  if (!socket) {
    socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 1000,
    });
  }
  return socket;
}

export function useWebSocket(symbols = []) {
  const socketRef = useRef(null);
  const addEvent = useMarketStore((state) => state.addEvent);
  const updateRealtimePrice = useMarketStore((state) => state.updateRealtimePrice);
  const setLatestSignal = useMarketStore((state) => state.addEvent);
  const updatePosition = useMarketStore((state) => state.updatePosition);

  useEffect(() => {
    socketRef.current = getSocket();
    const s = socketRef.current;

    // 连接成功
    s.on('connect', () => {
      console.log('WebSocket connected');
      addEvent({ type: 'system', message: '实时连接已建立' });

      // 订阅所有股票
      if (symbols.length > 0) {
        s.emit('subscribe', { symbols });
      }
    });

    // 断开连接
    s.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      addEvent({ type: 'warning', message: `实时连接已断开: ${reason}` });
    });

    // 连接错误
    s.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      addEvent({ type: 'error', message: '实时连接失败' });
    });

    // 价格更新
    s.on('price_update', (data) => {
      updateRealtimePrice(data.symbol, data.data);
    });

    // 信号
    s.on('signal', (data) => {
      setLatestSignal({
        type: 'signal',
        message: `信号: ${data.action} ${data.symbol} @ ${data.price}`,
        data,
      });
      addEvent({
        type: 'signal',
        message: `新信号: ${data.action} ${data.symbol}`,
        data,
      });
    });

    // 成交
    s.on('trade', (data) => {
      addEvent({
        type: 'trade',
        message: `成交: ${data.side.toUpperCase()} ${data.quantity} ${data.symbol} @ ${data.price}`,
        data,
      });
      // 更新持仓
      if (data.side === 'buy') {
        updatePosition(data.symbol, {
          quantity: data.quantity,
          price: data.price,
        });
      }
    });

    // 风控状态
    s.on('risk_status', (data) => {
      addEvent({
        type: 'risk',
        message: data.allowed ? '风控: 交易允许' : `风控: ${data.reason}`,
        data,
      });
    });

    // 交易器状态
    s.on('trader_status', (data) => {
      addEvent({
        type: 'system',
        message: `交易器: ${data.running ? '运行中' : '已停止'}`,
        data,
      });
    });

    return () => {
      // 取消订阅
      if (symbols.length > 0) {
        s.emit('unsubscribe', { symbols });
      }
      s.off('connect');
      s.off('disconnect');
      s.off('price_update');
      s.off('signal');
      s.off('trade');
      s.off('risk_status');
      s.off('trader_status');
    };
  }, [symbols.join(',')]);

  const subscribe = useCallback((newSymbols) => {
    if (socketRef.current) {
      socketRef.current.emit('subscribe', { symbols: newSymbols });
    }
  }, []);

  const unsubscribe = useCallback((targetSymbols) => {
    if (socketRef.current) {
      socketRef.current.emit('unsubscribe', { symbols: targetSymbols });
    }
  }, []);

  return {
    socket: socketRef.current,
    subscribe,
    unsubscribe,
  };
}

export default useWebSocket;
