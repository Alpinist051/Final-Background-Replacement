import type { EngineStats, ImageBackground, QualityUpdate, VirtualBackgroundTuning } from '@/types/engine';
import { loadImageBitmap } from '@/utils/canvasUtils';
import { DEFAULT_BACKGROUND } from '@/constants/defaultBackground';

type EngineCallbacks = {
  onStats?: (stats: EngineStats) => void;
  onError?: (error: string) => void;
  onStatus?: (status: 'idle' | 'starting' | 'running' | 'stopping' | 'error') => void;
};

type WorkerEnvelope =
  | { type: 'ready' }
  | { type: 'stats'; stats: EngineStats }
  | { type: 'frameProcessed' }
  | { type: 'quality'; quality: QualityUpdate }
  | {
      type: 'debug';
      step: string;
      message: string;
      metrics?: Record<string, string | number | boolean | null | undefined>;
    }
  | { type: 'error'; error: string };

type DebugCanvasesMessage = {
  type: 'debugCanvases';
  rawCanvas: OffscreenCanvas;
  cleanedCanvas: OffscreenCanvas;
};

type DebugMetrics = Record<string, string | number | boolean | null | undefined>;

function cloneBackgroundImage(background: ImageBackground): ImageBackground {
  return { mode: 'image', url: background.url, label: background.label };
}

function cloneTuning(tuning: VirtualBackgroundTuning): VirtualBackgroundTuning {
  return {
    temporalAlpha: tuning.temporalAlpha,
    bilateralSigmaSpatial: tuning.bilateralSigmaSpatial,
    bilateralSigmaColor: tuning.bilateralSigmaColor,
    feather: tuning.feather,
    lightWrap: tuning.lightWrap,
    confidenceBoost: tuning.confidenceBoost,
    motionBoost: tuning.motionBoost,
    brightnessBoost: tuning.brightnessBoost
  };
}

export class BackgroundEngine {
  private readonly videoElement: HTMLVideoElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly callbacks: EngineCallbacks;
  private worker: Worker | null = null;
  private cameraStream: MediaStream | null = null;
  private backgroundRevision = 0;
  private offscreenCanvas: OffscreenCanvas | null = null;
  private canvasTransferred = false;
  private debugRawCanvas: HTMLCanvasElement | null = null;
  private debugCleanedCanvas: HTMLCanvasElement | null = null;
  private debugRawOffscreenCanvas: OffscreenCanvas | null = null;
  private debugCleanedOffscreenCanvas: OffscreenCanvas | null = null;
  private debugCanvasesTransferred = false;
  private running = false;
  private inFlight = false;
  private queuedFrame = false;

  private tuning: VirtualBackgroundTuning = {
    temporalAlpha: 1,
    bilateralSigmaSpatial: 0,
    bilateralSigmaColor: 0,
    feather: 0,
    lightWrap: 0,
    confidenceBoost: 1.15,
    motionBoost: 0,
    brightnessBoost: 1
  };

  private background: ImageBackground = { ...DEFAULT_BACKGROUND };

  constructor(canvas: HTMLCanvasElement, callbacks: EngineCallbacks = {}) {
    this.canvas = canvas;
    this.callbacks = callbacks;
    this.videoElement = document.createElement('video');
    this.videoElement.autoplay = true;
    this.videoElement.muted = true;
    this.videoElement.playsInline = true;
    this.videoElement.style.display = 'none';
  }

  async startCapture(deviceId?: string) {
    this.callbacks.onStatus?.('starting');
    this.ensureWorker();

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: deviceId ? { exact: deviceId } : undefined,
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, min: 25 },
        facingMode: 'user'
      },
      audio: false
    });

    this.cameraStream = stream;
    this.videoElement.srcObject = stream;
    await this.videoElement.play();

    const width = this.videoElement.videoWidth || 1280;
    const height = this.videoElement.videoHeight || 720;

    this.logDebug('capture', 'Camera stream ready', {
      width,
      height,
      deviceId: deviceId ?? 'default'
    });

    if (!this.canvasTransferred) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.offscreenCanvas = this.canvas.transferControlToOffscreen();
      this.worker!.postMessage({
        type: 'init',
        canvas: this.offscreenCanvas,
        width,
        height,
        tuning: cloneTuning(this.tuning)
      }, [this.offscreenCanvas]);
      this.canvasTransferred = true;
    } else {
      this.worker!.postMessage({ type: 'resize', width, height });
      this.worker!.postMessage({ type: 'tuning', tuning: cloneTuning(this.tuning) });
    }

    await this.syncBackgroundToWorker();
    this.syncDebugCanvasesToWorker();

    this.running = true;
    this.callbacks.onStatus?.('running');
    this.scheduleNextPump();
  }

  async setBackground(background: ImageBackground) {
    this.background = cloneBackgroundImage(background);
    await this.syncBackgroundToWorker();
  }

  attachDebugCanvases(rawCanvas: HTMLCanvasElement, cleanedCanvas: HTMLCanvasElement) {
    if (this.debugRawCanvas === rawCanvas && this.debugCleanedCanvas === cleanedCanvas) {
      this.syncDebugCanvasesToWorker();
      return;
    }

    this.debugRawCanvas = rawCanvas;
    this.debugCleanedCanvas = cleanedCanvas;
    this.debugRawOffscreenCanvas = null;
    this.debugCleanedOffscreenCanvas = null;
    this.debugCanvasesTransferred = false;
    this.ensureDebugCanvasesTransferred();
    this.syncDebugCanvasesToWorker();
  }

  async stop() {
    this.running = false;
    this.inFlight = false;
    this.queuedFrame = false;
    this.callbacks.onStatus?.('stopping');
    this.cameraStream?.getTracks().forEach(t => t.stop());
    this.cameraStream = null;
    this.backgroundRevision += 1;
    this.callbacks.onStatus?.('idle');
  }

  dispose() {
    void this.stop();
    this.worker?.postMessage({ type: 'stop' });
    this.worker?.terminate();
    this.worker = null;
    this.backgroundRevision += 1;
  }

  private ensureWorker() {
    if (this.worker) return;
    this.worker = new Worker(new URL('../workers/processing.worker.ts', import.meta.url));

    this.worker.onmessage = (event: MessageEvent<WorkerEnvelope>) => {
      const msg = event.data;
      if (msg.type === 'stats') this.callbacks.onStats?.(msg.stats);
      if (msg.type === 'frameProcessed') {
        this.inFlight = false;
        if (this.queuedFrame) {
          this.queuedFrame = false;
          void this.pump();
        }
      }
      if (msg.type === 'ready') this.callbacks.onStatus?.('running');
      if (msg.type === 'error') {
        this.callbacks.onError?.(msg.error);
        this.callbacks.onStatus?.('error');
      }
      if (msg.type === 'debug') {
        this.logDebug(msg.step, msg.message, msg.metrics);
      }
      if (msg.type === 'quality') {
        this.tuning = { ...this.tuning, temporalAlpha: msg.quality.temporalAlpha };
      }
    };

    this.worker.onerror = (e) => {
      this.callbacks.onError?.(e.message);
      this.callbacks.onStatus?.('error');
    };
  }

  private ensureDebugCanvasesTransferred() {
    if (!this.debugRawCanvas || !this.debugCleanedCanvas) return;

    if (!this.debugRawOffscreenCanvas) {
      this.debugRawOffscreenCanvas = this.debugRawCanvas.transferControlToOffscreen();
    }

    if (!this.debugCleanedOffscreenCanvas) {
      this.debugCleanedOffscreenCanvas = this.debugCleanedCanvas.transferControlToOffscreen();
    }
  }

  private syncDebugCanvasesToWorker() {
    if (!this.worker || this.debugCanvasesTransferred) return;
    if (!this.debugRawOffscreenCanvas || !this.debugCleanedOffscreenCanvas) return;

    const message: DebugCanvasesMessage = {
      type: 'debugCanvases',
      rawCanvas: this.debugRawOffscreenCanvas,
      cleanedCanvas: this.debugCleanedOffscreenCanvas
    };

    this.worker.postMessage(message, [this.debugRawOffscreenCanvas, this.debugCleanedOffscreenCanvas]);
    this.debugCanvasesTransferred = true;
  }

  private async syncBackgroundToWorker() {
    if (!this.worker) return;

    const revision = ++this.backgroundRevision;

    try {
      const loadStart = performance.now();
      const bitmap = await loadImageBitmap(this.background.url);
      const width = bitmap.width;
      const height = bitmap.height;
      if (revision !== this.backgroundRevision || !this.worker) {
        bitmap.close();
        return;
      }
      this.worker.postMessage({ type: 'background', bitmap }, [bitmap]);
      this.logDebug('background', 'Background image loaded and sent to worker', {
        label: this.background.label ?? 'unnamed',
        width,
        height,
        loadMs: Math.round((performance.now() - loadStart) * 100) / 100
      });
    } catch (error) {
      this.callbacks.onError?.(error instanceof Error ? error.message : 'Failed to load background image');
    }
  }

  private async pump() {
    if (!this.running || !this.worker) return;
    if (this.inFlight) { this.queuedFrame = true; return; }

    if (this.videoElement.readyState < 2) {
      requestAnimationFrame(() => void this.pump());
      return;
    }

    this.inFlight = true;
    const frame = await createImageBitmap(this.videoElement);

    this.worker.postMessage(
      { type: 'frame', frame, timestamp: performance.now() },
      [frame]
    );

    this.scheduleNextPump();
  }

  private scheduleNextPump() {
    if (!this.running) return;
    if ('requestVideoFrameCallback' in this.videoElement) {
      this.videoElement.requestVideoFrameCallback(() => void this.pump());
    } else {
      requestAnimationFrame(() => void this.pump());
    }
  }

  private logDebug(step: string, message: string, metrics: DebugMetrics = {}) {
    const title = `[virtual-background][${step}] ${message}`;
    console.groupCollapsed(title);
    if (Object.keys(metrics).length > 0) {
      console.table(metrics);
    }
    console.groupEnd();
  }
}
