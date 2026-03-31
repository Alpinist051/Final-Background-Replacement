import { onBeforeUnmount, reactive, shallowRef } from 'vue';
import type { EngineState, ImageBackground } from '@/types/engine';
import { BackgroundEngine } from '@/core/BackgroundEngine';
import { DEFAULT_BACKGROUND } from '@/constants/defaultBackground';

function cloneBackgroundImage(background: ImageBackground): ImageBackground {
  return { mode: 'image', url: background.url, label: background.label };
}

export function useVirtualBackground() {
  const canvasRef = shallowRef<HTMLCanvasElement | null>(null);
  const debugRawCanvasRef = shallowRef<HTMLCanvasElement | null>(null);
  const debugCleanedCanvasRef = shallowRef<HTMLCanvasElement | null>(null);
  const engineRef = shallowRef<BackgroundEngine | null>(null);
  let currentBackground = cloneBackgroundImage(DEFAULT_BACKGROUND);

  const state = reactive<EngineState>({
    status: 'idle',
    error: null,
    stats: {
      fps: 0,
      latencyMs: 0,
      segmentationMs: 0,
      renderMs: 0,
      brightness: 0,
      motion: 0,
      droppedFrames: 0,
      processingWidth: 0,
      processingHeight: 0,
      foregroundRatio: 0,
      maskMean: 0,
      confidenceMean: 0
    }
  });

  function ensureEngine() {
    if (engineRef.value || !canvasRef.value) return engineRef.value;
    engineRef.value = new BackgroundEngine(canvasRef.value, {
      onStats: (stats) => { state.stats = stats; },
      onStatus: (status) => { state.status = status; },
      onError: (error) => {
        state.error = error;
        state.status = 'error';
      }
    });
    void engineRef.value.setBackground(currentBackground);
    return engineRef.value;
  }

  function syncDebugCanvases() {
    if (!engineRef.value || !debugRawCanvasRef.value || !debugCleanedCanvasRef.value) return;
    engineRef.value.attachDebugCanvases(debugRawCanvasRef.value, debugCleanedCanvasRef.value);
  }

  function attachCanvas(canvas: HTMLCanvasElement | null) {
    canvasRef.value = canvas;
    ensureEngine();
    syncDebugCanvases();
  }

  function attachDebugCanvases(rawCanvas: HTMLCanvasElement | null, cleanedCanvas: HTMLCanvasElement | null) {
    debugRawCanvasRef.value = rawCanvas;
    debugCleanedCanvasRef.value = cleanedCanvas;
    syncDebugCanvases();
  }

  async function start(deviceId?: string) {
    const engine = ensureEngine();
    if (!engine) throw new Error('Canvas not mounted');
    state.error = null;
    await engine.startCapture(deviceId);
  }

  async function stop() {
    await engineRef.value?.stop();
    state.status = 'idle';
  }

  async function setBackground(background: ImageBackground) {
    const nextBackground = cloneBackgroundImage(background);
    currentBackground = nextBackground;
    await engineRef.value?.setBackground(nextBackground);
  }

  function destroy() {
    void stop();
    engineRef.value?.dispose();
    engineRef.value = null;
  }

  onBeforeUnmount(() => destroy());

  return {
    state,
    canvasRef,
    attachCanvas,
    attachDebugCanvases,
    start,
    stop,
    setBackground,
    destroy
  };
}
