import { useEffect, useCallback, useRef } from 'react';
import React from 'react';
import { stocksAPI, riskAPI } from '../services/api';
import useMarketStore from '../stores/marketStore';

export function useWatchlist() {
  const watchlist = useMarketStore((state) => state.watchlist);
  const setWatchlist = useMarketStore((state) => state.setWatchlist);
  const setWatchlistLoading = useMarketStore((state) => state.setWatchlistLoading);
  const updateQuote = useMarketStore((state) => state.updateQuote);

  const fetchWatchlist = useCallback(async () => {
    setWatchlistLoading(true);
    try {
      const data = await stocksAPI.list();
      setWatchlist(data);
      return data;
    } catch (error) {
      if (!error.canceled) {
        console.error('Failed to fetch watchlist:', error);
      }
      return [];
    } finally {
      setWatchlistLoading(false);
    }
  }, [setWatchlist, setWatchlistLoading]);

  const refreshQuotes = useCallback(async (symbols) => {
    for (const symbol of symbols) {
      try {
        const quote = await stocksAPI.getQuote(symbol);
        updateQuote(symbol, quote);
      } catch (error) {
        if (!error.canceled) {
          console.error(`Failed to fetch quote for ${symbol}:`, error);
        }
      }
    }
  }, [updateQuote]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const fetchData = async () => {
      if (!cancelled) {
        await fetchWatchlist();
      }
    };

    fetchData();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [fetchWatchlist]);

  return {
    watchlist,
    fetchWatchlist,
    refreshQuotes,
  };
}

export function useBars(symbol, interval = '1d', days = 90, maPeriods = [5, 20, 50]) {
  const bars = useMarketStore((state) => state.bars[symbol]);
  const maData = useMarketStore((state) => state.maData[symbol]);
  const setBars = useMarketStore((state) => state.setBars);
  const setMAData = useMarketStore((state) => state.setMAData);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState(null);

  const abortControllerRef = useRef(null);
  const paramsRef = useRef({ symbol, interval, days, maPeriods });

  const fetchBars = useCallback(() => {
    // 取消之前的请求
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const params = paramsRef.current;
    if (!params.symbol) return;

    const controller = new AbortController();
    abortControllerRef.current = controller;

    const maString = params.maPeriods.join(',');
    setLoading(true);
    setError(null);

    stocksAPI.getBars(params.symbol, {
      interval: params.interval,
      days: params.days,
      ma: maString
    }, controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setBars(params.symbol, data.bars);
          setMAData(params.symbol, data.ma);
        }
      })
      .catch((err) => {
        if (!controller.signal.aborted && !err.canceled) {
          setError(err);
          console.error(`Failed to fetch bars for ${params.symbol}:`, err);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });
  }, [setBars, setMAData]);

  useEffect(() => {
    paramsRef.current = { symbol, interval, days, maPeriods };
    fetchBars();

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [symbol, interval, days, maPeriods.join(','), fetchBars]);

  return { bars, maData, loading, error, refetch: fetchBars };
}

export function useOrders() {
  const positions = useMarketStore((state) => state.positions);
  const trades = useMarketStore((state) => state.trades);
  const setPositions = useMarketStore((state) => state.setPositions);
  const setTrades = useMarketStore((state) => state.setTrades);

  const fetchOrders = useCallback(async () => {
    try {
      const data = await stocksAPI.list();
      setPositions(data.positions || {});
      const allTrades = [...(data.trades || [])];
      setTrades(allTrades);
    } catch (error) {
      if (!error.canceled) {
        console.error('Failed to fetch orders:', error);
      }
    }
  }, [setPositions, setTrades]);

  useEffect(() => {
    let cancelled = false;

    fetchOrders();
    const interval = setInterval(() => {
      if (!cancelled) {
        fetchOrders();
      }
    }, 30000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [fetchOrders]);

  return { positions, trades, refetch: fetchOrders };
}

export function usePerformance() {
  const performance = useMarketStore((state) => state.performance);
  const setPerformance = useMarketStore((state) => state.setPerformance);
  const riskStatus = useMarketStore((state) => state.riskStatus);
  const setRiskStatus = useMarketStore((state) => state.setRiskStatus);

  const fetchPerformance = useCallback(async () => {
    try {
      const perf = await riskAPI.getPerformance();
      setPerformance(perf);
    } catch (error) {
      if (!error.canceled) {
        console.error('Failed to fetch performance:', error);
      }
    }
  }, [setPerformance]);

  const fetchRiskStatus = useCallback(async () => {
    try {
      const status = await riskAPI.getStatus();
      setRiskStatus(status);
    } catch (error) {
      if (!error.canceled) {
        console.error('Failed to fetch risk status:', error);
      }
    }
  }, [setRiskStatus]);

  useEffect(() => {
    let cancelled = false;

    fetchPerformance();
    fetchRiskStatus();
    const interval = setInterval(() => {
      if (!cancelled) {
        fetchPerformance();
        fetchRiskStatus();
      }
    }, 30000);

    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [fetchPerformance, fetchRiskStatus]);

  return { performance, riskStatus, refetch: () => {
    fetchPerformance();
    fetchRiskStatus();
  }};
}

export default {
  useWatchlist,
  useBars,
  useOrders,
  usePerformance,
};
