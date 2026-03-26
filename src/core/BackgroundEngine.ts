import type { BackgroundMode, BackgroundSource, EngineStats, QualityUpdate, VirtualBackgroundTuning } from '@/types/engine';
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
    default:
      return background;
  }
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
  private backgroundVideo: HTMLVideoElement | null = null;
  private backgroundVideoTransferWarningShown = false;
  private backgroundRevision = 0;
  private offscreenCanvas: OffscreenCanvas | null = null;
  private canvasTransferred = false;
  private running = false;
  private inFlight = false;
  private queuedFrame = false;
  private processedStream: MediaStream | null = null;

  private tuning: VirtualBackgroundTuning = {
    temporalAlpha: 0.8,
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

    const width = this.videoElement.videoWidth || 1280;
    const height = this.videoElement.videoHeight || 720;

    this.processedStream ??= this.canvas.captureStream(30);

    if (!this.canvasTransferred) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.offscreenCanvas = this.canvas.transferControlToOffscreen();
      this.worker!.postMessage({
        type: 'init',
        canvas: this.offscreenCanvas,
        width,
        height,
        tuning: cloneTuning(this.tuning),
        background: cloneBackgroundSource(this.background)
      }, [this.offscreenCanvas]);
      this.canvasTransferred = true;
    } else {
      this.worker!.postMessage({ type: 'resize', width, height });
      this.worker!.postMessage({ type: 'tuning', tuning: cloneTuning(this.tuning) });
    }

    void this.syncBackgroundToWorker();

    this.running = true;
    this.callbacks.onStatus?.('running');
    this.scheduleNextPump();
  }

  setTuning(tuning: VirtualBackgroundTuning) {
    this.tuning = cloneTuning(tuning);
    this.worker?.postMessage({ type: 'tuning', tuning: cloneTuning(this.tuning) });
  }

  async setBackground(background: BackgroundSource) {
    this.background = cloneBackgroundSource(background);
    await this.syncBackgroundToWorker();
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
    this.stopBackgroundVideo();
    this.backgroundRevision += 1;
    this.callbacks.onStatus?.('idle');
  }

  dispose() {
    this.stop();
    this.worker?.postMessage({ type: 'stop' });
    this.worker?.terminate();
    this.worker = null;
    this.processedStream = null;
    this.backgroundRevision += 1;
  }

  private resizeCanvas() {
    if (this.canvasTransferred) return;
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

  private stopBackgroundVideo() {
    this.backgroundVideo?.pause();
    this.backgroundVideo = null;
    this.backgroundVideoTransferWarningShown = false;
  }

  private async syncBackgroundToWorker() {
    if (!this.worker) return;

    const revision = ++this.backgroundRevision;
    const background = cloneBackgroundSource(this.background);

    switch (background.mode) {
      case 'solid':
      case 'blur':
        this.stopBackgroundVideo();
        this.worker.postMessage({ type: 'background', background });
        return;
      case 'video': {
        this.stopBackgroundVideo();
        const video = document.createElement('video');
        video.crossOrigin = 'anonymous';
        video.src = background.url;
        video.loop = background.loop ?? true;
        video.autoplay = true;
        video.muted = true;
        video.playsInline = true;
        await video.play().catch(() => { });
        if (revision !== this.backgroundRevision || !this.worker) {
          video.pause();
          return;
        }
        this.backgroundVideo = video;
        this.worker.postMessage({ type: 'background', background });
        return;
      }
      case 'image':
        try {
          this.stopBackgroundVideo();
          const bitmap = await loadImageBitmap(background.url);
          if (revision !== this.backgroundRevision || !this.worker) {
            bitmap.close();
            return;
          }
          this.worker.postMessage({ type: 'background', background, bitmap }, [bitmap]);
        } catch (error) {
          this.callbacks.onError?.(error instanceof Error ? error.message : 'Failed to load background image');
        }
        return;
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
    let backgroundFrame: ImageBitmap | null = null;

    if (this.background.mode === 'video') {
      const backgroundVideo = this.backgroundVideo;
      if (backgroundVideo && backgroundVideo.readyState >= 2) {
        try {
          backgroundFrame = await createImageBitmap(backgroundVideo);
        } catch (error) {
          if (!this.backgroundVideoTransferWarningShown) {
            this.backgroundVideoTransferWarningShown = true;
            console.warn('Video background is not origin-clean, so it cannot be transferred to the worker. Use a CORS-enabled video source or a local file for animated backgrounds.', error);
          }
        }
      }
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
