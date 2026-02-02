import { useEffect, useCallback } from 'react';
import { annotationsAPI } from '../services/api';
import useAnnotationStore from '../stores/annotationStore';

export function useAnnotations(symbol) {
  const markers = useAnnotationStore((state) => state.markers[symbol] || []);
  const sellRanges = useAnnotationStore((state) => state.sellRanges[symbol] || []);
  const loading = useAnnotationStore((state) => state.loading);
  const setLoading = useAnnotationStore((state) => state.setLoading);
  const setMarkers = useAnnotationStore((state) => state.setMarkers);
  const addMarker = useAnnotationStore((state) => state.addMarker);
  const updateMarker = useAnnotationStore((state) => state.updateMarker);
  const removeMarker = useAnnotationStore((state) => state.removeMarker);
  const setSellRanges = useAnnotationStore((state) => state.setSellRanges);
  const addSellRange = useAnnotationStore((state) => state.addSellRange);
  const updateSellRange = useAnnotationStore((state) => state.updateSellRange);
  const removeSellRange = useAnnotationStore((state) => state.removeSellRange);

  const fetchAnnotations = useCallback(async () => {
    if (!symbol) return;
    setLoading(true);
    try {
      const [markerData, rangeData] = await Promise.all([
        annotationsAPI.getMarkers(symbol),
        annotationsAPI.getRanges(symbol),
      ]);
      setMarkers(symbol, markerData);
      setSellRanges(symbol, rangeData);
    } catch (error) {
      console.error('Failed to fetch annotations:', error);
    } finally {
      setLoading(false);
    }
  }, [symbol, setMarkers, setSellRanges, setLoading]);

  useEffect(() => {
    fetchAnnotations();
  }, [fetchAnnotations]);

  // 标注操作
  const createMarker = useCallback(async (data) => {
    try {
      const marker = await annotationsAPI.addMarker(symbol, data);
      addMarker(symbol, marker);
      return marker;
    } catch (error) {
      console.error('Failed to create marker:', error);
      throw error;
    }
  }, [symbol, addMarker]);

  const editMarker = useCallback(async (markerId, data) => {
    try {
      const marker = await annotationsAPI.updateMarker(symbol, markerId, data);
      updateMarker(symbol, markerId, marker);
      return marker;
    } catch (error) {
      console.error('Failed to update marker:', error);
      throw error;
    }
  }, [symbol, updateMarker]);

  const deleteMarker = useCallback(async (markerId) => {
    try {
      await annotationsAPI.deleteMarker(symbol, markerId);
      removeMarker(symbol, markerId);
    } catch (error) {
      console.error('Failed to delete marker:', error);
      throw error;
    }
  }, [symbol, removeMarker]);

  // 卖出区间操作
  const createSellRange = useCallback(async (data) => {
    try {
      const range = await annotationsAPI.addRange(symbol, data);
      addSellRange(symbol, range);
      return range;
    } catch (error) {
      console.error('Failed to create sell range:', error);
      throw error;
    }
  }, [symbol, addSellRange]);

  const editSellRange = useCallback(async (rangeId, data) => {
    try {
      const range = await annotationsAPI.updateRange(symbol, rangeId, data);
      updateSellRange(symbol, rangeId, range);
      return range;
    } catch (error) {
      console.error('Failed to update sell range:', error);
      throw error;
    }
  }, [symbol, updateSellRange]);

  const deleteSellRange = useCallback(async (rangeId) => {
    try {
      await annotationsAPI.deleteRange(symbol, rangeId);
      removeSellRange(symbol, rangeId);
    } catch (error) {
      console.error('Failed to delete sell range:', error);
      throw error;
    }
  }, [symbol, removeSellRange]);

  return {
    markers,
    sellRanges,
    loading,
    refetch: fetchAnnotations,
    // 标注操作
    createMarker,
    editMarker,
    deleteMarker,
    // 卖出区间操作
    createSellRange,
    editSellRange,
    deleteSellRange,
  };
}

export default useAnnotations;
