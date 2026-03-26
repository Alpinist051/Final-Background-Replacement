import { computed, onBeforeUnmount, onMounted, reactive, shallowRef, watch } from 'vue';
import type { BackgroundSource, EngineState, VirtualBackgroundTuning } from '@/types/engine';
import { BackgroundEngine } from '@/core/BackgroundEngine';

const defaultTuning: VirtualBackgroundTuning = {
  temporalAlpha: 0.82,
  bilateralSigmaSpatial: 4,
  bilateralSigmaColor: 0.1,
  feather: 0.08,
  lightWrap: 0.15,
  confidenceBoost: 1.2,
  motionBoost: 1,
  brightnessBoost: 1.3
};

const defaultBackground: BackgroundSource = { mode: 'solid', color: '#111827' };

export function useVirtualBackground() {
  const canvasRef = shallowRef<HTMLCanvasElement | null>(null);
  const engineRef = shallowRef<BackgroundEngine | null>(null);

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
      processingHeight: 0
    },
    tuning: { ...defaultTuning },
    background: defaultBackground
  });

  const processedTrack = computed(() => engineRef.value?.getProcessedTrack() ?? null);

  function ensureEngine() {
    if (engineRef.value || !canvasRef.value) return engineRef.value;
    engineRef.value = new BackgroundEngine(canvasRef.value, {
      onStats: (stats) => { state.stats = stats; },
      onStatus: (status) => { state.status = status; },
      onQuality: (quality) => {
        state.tuning = { ...state.tuning, temporalAlpha: quality.temporalAlpha };
      },
      onError: (error) => {
        state.error = error;
        state.status = 'error';
      }
    });
    return engineRef.value;
  }

  function attachCanvas(canvas: HTMLCanvasElement | null) {
    canvasRef.value = canvas;
    ensureEngine();
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

  function setTuning(next: Partial<VirtualBackgroundTuning>) {
    state.tuning = { ...state.tuning, ...next };
    engineRef.value?.setTuning(state.tuning);
  }

  async function setBackground(background: BackgroundSource) {
    state.background = background;
    await engineRef.value?.setBackground(background);
  }

  function destroy() {
    void stop();
    engineRef.value?.dispose();
    engineRef.value = null;
  }

  // Auto-cleanup when component unmounts
  onMounted(() => ensureEngine());
  onBeforeUnmount(() => destroy());

  return {
    state,
    canvasRef,
    processedTrack,
    attachCanvas,
    start,
    stop,
    setTuning,
    setBackground,
    destroy
  };
}