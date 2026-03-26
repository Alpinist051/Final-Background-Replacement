import { createAnalysisCanvas } from '@/utils/canvasUtils';
import { createPerformanceTracker } from '@/utils/performance';
import type { BackgroundSource, EngineStats, VirtualBackgroundTuning } from '@/types/engine';
import { SegmentationManager } from '@/core/SegmentationManager';
import { MaskProcessor } from '@/core/MaskProcessor';
import { WebGLRenderer, type RenderFrameArgs } from '@/core/WebGLRenderer';

type InitMessage = {
  type: 'init';
  canvas: OffscreenCanvas;
  width: number;
  height: number;
  tuning: VirtualBackgroundTuning;
  background: BackgroundSource;
};

type FrameMessage = {
  type: 'frame';
  frame: ImageBitmap;
  timestamp: number;
  backgroundFrame?: ImageBitmap | null;
};

type UpdateTuningMessage = {
  type: 'tuning';
  tuning: VirtualBackgroundTuning;
};

type UpdateBackgroundMessage = {
  type: 'background';
  background: BackgroundSource;
  bitmap?: ImageBitmap;
};

type ResizeMessage = {
  type: 'resize';
  width: number;
  height: number;
};

type WorkerMessage = InitMessage | FrameMessage | UpdateTuningMessage | UpdateBackgroundMessage | ResizeMessage | { type: 'stop' };

let renderer: WebGLRenderer | null = null;
let segmenter: SegmentationManager | null = null;
let maskProcessor: MaskProcessor | null = null;
let analysisCanvas: ReturnType<typeof createAnalysisCanvas> | null = null;
let processingCanvas: OffscreenCanvas | null = null;
let processingContext: OffscreenCanvasRenderingContext2D | null = null;
let currentTuning: VirtualBackgroundTuning = {
  temporalAlpha: 0.82,
  bilateralSigmaSpatial: 4,
  bilateralSigmaColor: 0.1,
  feather: 0.08,
  lightWrap: 0.15,
  confidenceBoost: 1.2,
  motionBoost: 1,
  brightnessBoost: 1
};
let currentBackground: BackgroundSource = { mode: 'solid', color: '#111827' };
const performanceTracker = createPerformanceTracker();
let previousLuma: Float32Array | null = null;
let pendingFrame: ImageBitmap | null = null;
let pendingBackgroundFrame: ImageBitmap | null = null;
let processingWidth = 1280;
let processingHeight = 720;
let degraded = false;
let lowFpsStreak = 0;
let tickHandle: number | null = null;
let lastMaskWarningAt = 0;
const TARGET_FPS = 30;
const LOW_FPS_THRESHOLD = 25;
const LOW_FPS_FRAMES_BEFORE_DROP = 5;

function closeBitmap(bitmap: ImageBitmap | null) {
  bitmap?.close();
}

function scheduleTick() {
  if (tickHandle !== null) return;

  const tick = () => {
    tickHandle = null;
    void processTick();
  };

  if (typeof self.requestAnimationFrame === 'function') {
    tickHandle = self.requestAnimationFrame(tick);
    return;
  }

  tickHandle = self.setTimeout(tick, 1000 / TARGET_FPS);
}

function ensureProcessingCanvas() {
  if (!processingCanvas) {
    processingCanvas = new OffscreenCanvas(processingWidth, processingHeight);
    processingContext = processingCanvas.getContext('2d', { willReadFrequently: true });
    return;
  }

  if (processingCanvas.width !== processingWidth || processingCanvas.height !== processingHeight) {
    processingCanvas.width = processingWidth;
    processingCanvas.height = processingHeight;
    processingContext = processingCanvas.getContext('2d', { willReadFrequently: true });
  }
}

async function drawForProcessing(bitmap: ImageBitmap) {
  ensureProcessingCanvas();
  if (!processingCanvas || !processingContext) {
    return bitmap;
  }

  processingContext.clearRect(0, 0, processingCanvas.width, processingCanvas.height);
  // Rotate 180 degrees so upside-down camera sensors render upright in both
  // segmentation and the final composite.
  processingContext.setTransform(-1, 0, 0, -1, processingCanvas.width, processingCanvas.height);
  processingContext.drawImage(bitmap, 0, 0, processingCanvas.width, processingCanvas.height);
  processingContext.setTransform(1, 0, 0, 1, 0, 0);
  return createImageBitmap(processingCanvas);
}

function computeLuma(bitmap: ImageBitmap) {
  analysisCanvas ??= createAnalysisCanvas(32, 18);
  const { canvas, context } = analysisCanvas;
  if (!context) {
    return { brightness: 0, motion: 0 };
  }

  canvas.width = 32;
  canvas.height = 18;
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
  const current = new Float32Array(canvas.width * canvas.height);
  let brightness = 0;

  for (let i = 0; i < current.length; i += 1) {
    const index = i * 4;
    const value = (data[index] * 0.299 + data[index + 1] * 0.587 + data[index + 2] * 0.114) / 255;
    current[i] = value;
    brightness += value;
  }

  brightness = (brightness / current.length) * 255;

  let motion = 0;
  if (previousLuma && previousLuma.length === current.length) {
    let delta = 0;
    for (let i = 0; i < current.length; i += 1) {
      delta += Math.abs(current[i] - previousLuma[i]);
    }
    motion = delta / current.length;
  }
  previousLuma = current;
  return { brightness, motion };
}

function boostTuning(brightness: number, motion: number) {
  const boosted = { ...currentTuning };
  if (motion > 0.3) {
    boosted.confidenceBoost = Math.max(boosted.confidenceBoost, 1.5) * boosted.motionBoost;
  }
  if (brightness < 80) {
    boosted.confidenceBoost *= boosted.brightnessBoost * 1.3;
  }
  boosted.confidenceBoost = Math.min(2.5, boosted.confidenceBoost);
  boosted.temporalAlpha = Math.min(0.9, Math.max(0.75, boosted.temporalAlpha));
  return boosted;
}

function applyQualityFallback(fps: number) {
  if (fps < LOW_FPS_THRESHOLD) {
    lowFpsStreak += 1;
  } else {
    lowFpsStreak = 0;
  }

  if (!degraded && lowFpsStreak >= LOW_FPS_FRAMES_BEFORE_DROP) {
    degraded = true;
    processingWidth = 640;
    processingHeight = 480;
    currentTuning = {
      ...currentTuning,
      temporalAlpha: 0.9
    };
    renderer?.resize(processingWidth, processingHeight);
    postMessage({
      type: 'quality',
      quality: {
        width: processingWidth,
        height: processingHeight,
        temporalAlpha: currentTuning.temporalAlpha
      }
    });
  }
}

async function handleInit(message: InitMessage) {
  renderer = new WebGLRenderer(message.canvas);
  processingWidth = message.width;
  processingHeight = message.height;
  renderer.resize(processingWidth, processingHeight);
  renderer.setBackground(message.background);
  segmenter = new SegmentationManager();
  await segmenter.initialize();
  maskProcessor = new MaskProcessor();
  currentTuning = message.tuning;
  currentBackground = message.background;
  scheduleTick();
  postMessage({ type: 'ready' });
}

async function handleFrame(message: FrameMessage) {
  closeBitmap(pendingFrame);
  closeBitmap(pendingBackgroundFrame);
  pendingFrame = message.frame;
  pendingBackgroundFrame = message.backgroundFrame ?? null;
  scheduleTick();
}

async function processTick() {
  if (!renderer || !segmenter || !maskProcessor || !pendingFrame) {
    scheduleTick();
    return;
  }

  const frameStart = performance.now();
  const sourceFrame = pendingFrame;
  const sourceBackgroundFrame = pendingBackgroundFrame;
  pendingFrame = null;
  pendingBackgroundFrame = null;

  const processedBitmap = await drawForProcessing(sourceFrame);
  const { brightness, motion } = computeLuma(processedBitmap);
  const tuning = boostTuning(brightness, motion);

  const segmentationStart = performance.now();
  const segmentation = await segmenter.segment(processedBitmap, Math.round(frameStart));
  const segmentationMs = performance.now() - segmentationStart;
  const processedMask = maskProcessor.process(segmentation, tuning);
  const combinedMotion = Math.max(motion, processedMask.motionMagnitude);
  const renderStart = performance.now();

  if ((processedMask.foregroundRatio < 0.01 || processedMask.foregroundRatio > 0.99) && performance.now() - lastMaskWarningAt > 3000) {
    lastMaskWarningAt = performance.now();
    console.warn(
      `Foreground mask looks suspicious (${(processedMask.foregroundRatio * 100).toFixed(1)}% coverage, mean ${processedMask.maskMean.toFixed(2)}).`
    );
  }

  const renderArgs: RenderFrameArgs = {
    frame: processedBitmap,
    alphaMask: processedMask.alphaMask,
    confidenceMask: processedMask.confidenceMask,
    backgroundFrame: sourceBackgroundFrame,
    background: currentBackground,
    tuning
  };

  await renderer.renderFrame(renderArgs);
  const renderMs = performance.now() - renderStart;
  const latencyMs = performance.now() - frameStart;
  const fps = latencyMs > 0 ? 1000 / latencyMs : 0;

  performanceTracker.record({
    fps,
    latencyMs,
    segmentationMs,
    renderMs,
    brightness,
    motion: combinedMotion,
    droppedFrames: 0,
    processingWidth,
    processingHeight,
    foregroundRatio: processedMask.foregroundRatio,
    maskMean: processedMask.maskMean,
    confidenceMean: processedMask.confidenceMean
  });

  applyQualityFallback(fps);

  postMessage({
    type: 'stats',
    stats: performanceTracker.snapshot() satisfies EngineStats
  });

  if (processedBitmap !== sourceFrame) {
    processedBitmap.close();
  }
  sourceFrame.close();
  sourceBackgroundFrame?.close?.();
  postMessage({ type: 'frameProcessed' });
  scheduleTick();
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  if (message.type === 'init') {
    await handleInit(message);
    return;
  }
  if (message.type === 'frame') {
    await handleFrame(message);
    return;
  }
  if (message.type === 'tuning') {
    currentTuning = message.tuning;
    return;
  }
  if (message.type === 'background') {
    currentBackground = message.background;
    renderer?.setBackground(message.background);
    if (message.background.mode === 'image' && message.bitmap) {
      renderer?.setBackgroundBitmap(message.bitmap);
      message.bitmap.close();
    }
    return;
  }
  if (message.type === 'resize') {
    processingWidth = message.width;
    processingHeight = message.height;
    if (!degraded) {
      renderer?.resize(processingWidth, processingHeight);
    }
    return;
  }
  if (message.type === 'stop') {
    if (tickHandle !== null) {
      if (typeof self.cancelAnimationFrame === 'function') {
        self.cancelAnimationFrame(tickHandle);
      } else {
        self.clearTimeout(tickHandle);
      }
      tickHandle = null;
    }
    closeBitmap(pendingFrame);
    closeBitmap(pendingBackgroundFrame);
    pendingFrame = null;
    pendingBackgroundFrame = null;
    segmenter?.close();
    maskProcessor?.reset();
    renderer?.destroy();
    renderer = null;
    segmenter = null;
    maskProcessor = null;
    previousLuma = null;
    degraded = false;
    lowFpsStreak = 0;
    lastMaskWarningAt = 0;
  }
};
