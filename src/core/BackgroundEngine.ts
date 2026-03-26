import type { BackgroundSource, EngineStats, VirtualBackgroundTuning, QualityUpdate } from '@/types/engine';
import { loadImageBitmap } from '@/utils/canvasUtils';

type EngineCallbacks = {
  onStats?: (stats: EngineStats) => void;
  onError?: (error: string) => void;
  onStatus?: (status: 'idle' | 'starting' | 'running' | 'stopping' | 'error') => void;
  onQuality?: (quality: QualityUpdate) => void;
};

type WorkerEnvelope =
  | { type: 'ready' }
  | { type: 'stats'; stats: EngineStats }
  | { type: 'frameProcessed' }
  | { type: 'quality'; quality: QualityUpdate }
  | { type: 'error'; error: string };

function cloneBackgroundSource(background: BackgroundSource): BackgroundSource {
  switch (background.mode) {
    case 'solid':
      return { mode: 'solid', color: background.color };
    case 'image':
      return { mode: 'image', url: background.url, label: background.label };
    case 'video':
      return { mode: 'video', url: background.url, label: background.label, loop: background.loop };
    case 'blur':
      return { mode: 'blur', strength: background.strength };
  }
}

export class BackgroundEngine {
  private readonly videoElement: HTMLVideoElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly callbacks: EngineCallbacks;
  private worker: Worker | null = null;
  private cameraStream: MediaStream | null = null;
  private backgroundVideo: HTMLVideoElement | null = null;
  private running = false;
  private inFlight = false;
  private queuedFrame = false;
  private processedStream: MediaStream | null = null;
  private initialized = false;

  private tuning: VirtualBackgroundTuning = {
    temporalAlpha: 0.82,
    bilateralSigmaSpatial: 4,
    bilateralSigmaColor: 0.1,
    feather: 0.08,
    lightWrap: 0.15,
    confidenceBoost: 1.2,
    motionBoost: 1,
    brightnessBoost: 1.3
  };

  private background: BackgroundSource = { mode: 'solid', color: '#111827' };

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

    this.resizeCanvas();

    this.processedStream ??= this.canvas.captureStream(30);

    const offscreen = this.canvas.transferControlToOffscreen();
    this.worker!.postMessage({
      type: 'init',
      canvas: offscreen,
      width: this.canvas.width,
      height: this.canvas.height,
      tuning: this.tuning,
      background: cloneBackgroundSource(this.background)
    }, [offscreen]);

    this.initialized = true;
    this.running = true;
    this.callbacks.onStatus?.('running');
    this.scheduleNextPump();
  }

  setTuning(tuning: VirtualBackgroundTuning) {
    this.tuning = tuning;
    this.worker?.postMessage({ type: 'tuning', tuning });
  }

  async setBackground(background: BackgroundSource) {
    this.background = cloneBackgroundSource(background);
    if (!this.worker) return;

    if (this.background.mode === 'image') {
      const bitmap = await loadImageBitmap(this.background.url);
      this.worker.postMessage({ type: 'background', background: cloneBackgroundSource(this.background), bitmap }, [bitmap]);
      return;
    }

    if (this.background.mode === 'video') {
      this.backgroundVideo?.pause();
      this.backgroundVideo = null;
      const video = document.createElement('video');
      video.src = this.background.url;
      video.loop = this.background.loop ?? true;
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      await video.play().catch(() => { });
      this.backgroundVideo = video;
    }

    this.worker.postMessage({ type: 'background', background: cloneBackgroundSource(this.background) });
  }

  getProcessedTrack() {
    return this.processedStream?.getVideoTracks()[0] ?? null;
  }

  async stop() {
    this.running = false;
    this.inFlight = false;
    this.queuedFrame = false;
    this.callbacks.onStatus?.('stopping');
    this.cameraStream?.getTracks().forEach(t => t.stop());
    this.cameraStream = null;
    this.backgroundVideo?.pause();
    this.backgroundVideo = null;
    this.callbacks.onStatus?.('idle');
  }

  dispose() {
    this.stop();
    this.worker?.postMessage({ type: 'stop' });
    this.worker?.terminate();
    this.worker = null;
    this.initialized = false;
    this.processedStream = null;
  }

  private resizeCanvas() {
    this.canvas.width = this.videoElement.videoWidth || 1280;
    this.canvas.height = this.videoElement.videoHeight || 720;
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
      if (msg.type === 'quality') {
        this.tuning = { ...this.tuning, temporalAlpha: msg.quality.temporalAlpha };
        this.callbacks.onQuality?.(msg.quality);
      }
    };

    this.worker.onerror = (e) => {
      this.callbacks.onError?.(e.message);
      this.callbacks.onStatus?.('error');
    };
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
    let backgroundFrame: ImageBitmap | null = null;

    if (this.background.mode === 'video' && this.backgroundVideo?.readyState >= 2) {
      backgroundFrame = await createImageBitmap(this.backgroundVideo);
    }

    this.worker.postMessage(
      { type: 'frame', frame, timestamp: performance.now(), backgroundFrame },
      backgroundFrame ? [frame, backgroundFrame] : [frame]
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
}
