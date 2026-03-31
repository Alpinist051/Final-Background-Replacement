import { createAnalysisCanvas } from '@/utils/canvasUtils';
import { createPerformanceTracker } from '@/utils/performance';
import type { EngineStats, VirtualBackgroundTuning } from '@/types/engine';
import { SegmentationManager } from '@/core/SegmentationManager';
import { MaskProcessor } from '@/core/MaskProcessor';
import { WebGLRenderer, type RenderFrameArgs } from '@/core/WebGLRenderer';

type InitMessage = {
  type: 'init';
  canvas: OffscreenCanvas;
  width: number;
  height: number;
  tuning: VirtualBackgroundTuning;
};

type FrameMessage = {
  type: 'frame';
  frame: ImageBitmap;
  timestamp: number;
};

type UpdateTuningMessage = {
  type: 'tuning';
  tuning: VirtualBackgroundTuning;
};

type UpdateBackgroundMessage = {
  type: 'background';
  bitmap: ImageBitmap;
};

type DebugCanvasesMessage = {
  type: 'debugCanvases';
  rawCanvas: OffscreenCanvas;
  cleanedCanvas: OffscreenCanvas;
};

type DebugStep = 'init' | 'preprocess' | 'segmentation' | 'mask' | 'background' | 'composite';

type DebugMetrics = Record<string, string | number | boolean | null | undefined>;

type DebugMessage = {
  type: 'debug';
  step: DebugStep;
  message: string;
  metrics?: DebugMetrics;
};

type ResizeMessage = {
  type: 'resize';
  width: number;
  height: number;
};

type WorkerMessage = InitMessage | FrameMessage | UpdateTuningMessage | UpdateBackgroundMessage | DebugCanvasesMessage | ResizeMessage | { type: 'stop' };

let renderer: WebGLRenderer | null = null;
let segmenter: SegmentationManager | null = null;
let maskProcessor: MaskProcessor | null = null;
let analysisCanvas: ReturnType<typeof createAnalysisCanvas> | null = null;
let processingCanvas: OffscreenCanvas | null = null;
let processingContext: OffscreenCanvasRenderingContext2D | null = null;
let debugRawCanvas: OffscreenCanvas | null = null;
let debugRawContext: OffscreenCanvasRenderingContext2D | null = null;
let debugCleanedCanvas: OffscreenCanvas | null = null;
let debugCleanedContext: OffscreenCanvasRenderingContext2D | null = null;

type QualityTier = {
  maxWidth: number;
  maxHeight: number;
};

const QUALITY_TIERS: QualityTier[] = [
  { maxWidth: 1280, maxHeight: 720 },
  { maxWidth: 960, maxHeight: 540 },
  { maxWidth: 768, maxHeight: 432 },
  { maxWidth: 640, maxHeight: 360 }
];

let currentTuning: VirtualBackgroundTuning = {
  temporalAlpha: 1,
  bilateralSigmaSpatial: 0,
  bilateralSigmaColor: 0,
  feather: 0,
  lightWrap: 0,
  confidenceBoost: 1.15,
  motionBoost: 0,
  brightnessBoost: 1
};
const performanceTracker = createPerformanceTracker();
let previousLuma: Float32Array | null = null;
let pendingFrame: ImageBitmap | null = null;
let sourceWidth = 1280;
let sourceHeight = 720;
let processingWidth = 640;
let processingHeight = 480;
let qualityTierIndex = 0;
let tickHandle: number | null = null;
let lastMaskWarningAt = 0;
let lastDebugAt: Partial<Record<DebugStep, number>> = {};
const TARGET_FPS = 30;
const LOW_FPS_THRESHOLD = 20;
const LOW_FPS_FRAMES_BEFORE_DROP = 8;
const HIGH_FPS_FRAMES_BEFORE_RAISE = 18;
const DEBUG_LOG_INTERVAL_MS = 2000;

function closeBitmap(bitmap: ImageBitmap | null) {
  bitmap?.close();
}

function fitWithinBounds(width: number, height: number, maxWidth: number, maxHeight: number) {
  const scale = Math.min(maxWidth / width, maxHeight / height, 1);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale))
  };
}

function updateProcessingResolution(tierIndex: number, announce = false) {
  qualityTierIndex = Math.max(0, Math.min(QUALITY_TIERS.length - 1, tierIndex));
  const tier = QUALITY_TIERS[qualityTierIndex];
  const fit = fitWithinBounds(sourceWidth, sourceHeight, tier.maxWidth, tier.maxHeight);
  processingWidth = fit.width;
  processingHeight = fit.height;
  renderer?.resize(processingWidth, processingHeight);
  maskProcessor?.reset();
  if (announce) {
    postMessage({
      type: 'quality',
      quality: { width: processingWidth, height: processingHeight, temporalAlpha: currentTuning.temporalAlpha }
    });
  }
}

function scheduleTick() {
  if (tickHandle !== null) return;
  const tick = () => { tickHandle = null; void processTick(); };
  if (typeof self.requestAnimationFrame === 'function') {
    tickHandle = self.requestAnimationFrame(tick);
  } else {
    tickHandle = self.setTimeout(tick, 1000 / TARGET_FPS);
  }
}

function ensureProcessingCanvas() {
  if (!processingCanvas || processingCanvas.width !== processingWidth || processingCanvas.height !== processingHeight) {
    processingCanvas = new OffscreenCanvas(processingWidth, processingHeight);
    processingContext = processingCanvas.getContext('2d', { willReadFrequently: true });
  }
}

function drawMaskPreview(
  canvas: OffscreenCanvas | null,
  context: OffscreenCanvasRenderingContext2D | null,
  mask: Float32Array,
  width: number,
  height: number
) {
  if (!canvas) return context;

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const nextContext = context ?? canvas.getContext('2d', { willReadFrequently: true, alpha: false });
  if (!nextContext) return context;

  const imageData = nextContext.createImageData(width, height);
  const pixels = imageData.data;

  for (let i = 0; i < mask.length; i += 1) {
    const shade = Math.max(0, Math.min(255, Math.round((mask[i] ?? 0) * 255)));
    const pixelIndex = i * 4;
    pixels[pixelIndex + 0] = shade;
    pixels[pixelIndex + 1] = shade;
    pixels[pixelIndex + 2] = shade;
    pixels[pixelIndex + 3] = 255;
  }

  nextContext.putImageData(imageData, 0, 0);
  return nextContext;
}

function roundMetric(value: number) {
  return Math.round(value * 100) / 100;
}

function summarizeMask(mask: ArrayLike<number>, threshold = 0.5) {
  const length = mask.length;
  if (length === 0) {
    return { mean: 0, coverage: 0, min: 0, max: 0 };
  }

  let sum = 0;
  let coverage = 0;
  let min = 1;
  let max = 0;

  for (let i = 0; i < length; i += 1) {
    const value = Math.max(0, Math.min(1, mask[i] ?? 0));
    sum += value;
    if (value >= threshold) coverage += 1;
    if (value < min) min = value;
    if (value > max) max = value;
  }

  return {
    mean: sum / length,
    coverage: coverage / length,
    min,
    max
  };
}

function postDebug(step: DebugStep, message: string, metrics: DebugMetrics = {}, force = false) {
  const now = performance.now();
  const last = lastDebugAt[step] ?? -Infinity;
  if (!force && now - last < DEBUG_LOG_INTERVAL_MS) return;
  lastDebugAt[step] = now;
  const payload: DebugMessage = { type: 'debug', step, message, metrics };
  postMessage(payload);
}

async function drawForProcessing(bitmap: ImageBitmap) {
  ensureProcessingCanvas();
  if (!processingCanvas || !processingContext) return bitmap;
  processingContext.clearRect(0, 0, processingCanvas.width, processingCanvas.height);
  processingContext.drawImage(bitmap, 0, 0, processingCanvas.width, processingCanvas.height);
  processingContext.setTransform(1, 0, 0, 1, 0, 0);
  return createImageBitmap(processingCanvas);
}

function computeLuma(bitmap: ImageBitmap) {
  analysisCanvas ??= createAnalysisCanvas(32, 18);
  const { canvas, context } = analysisCanvas;
  if (!context) return { brightness: 0, motion: 0 };

  canvas.width = 32; canvas.height = 18;
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
    let peakDelta = 0;
    let activePixels = 0;
    for (let i = 0; i < current.length; i += 1) {
      const diff = Math.abs(current[i] - previousLuma[i]);
      delta += diff;
      if (diff > peakDelta) peakDelta = diff;
      if (diff > 0.035) activePixels += 1;
    }
    const meanDelta = delta / current.length;
    const activeCoverage = activePixels / current.length;
    // Localized motion matters more than global brightness drift for mask updates.
    motion = Math.min(1, Math.max(meanDelta, peakDelta * 0.5, activeCoverage * 0.85));
  }
  previousLuma = current;
  return { brightness, motion };
}

function applyQualityFallback(fps: number, segmentationMs: number) {
  void fps;
  void segmentationMs;
  // Keep the highest processing resolution for human detection quality.
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

async function handleInit(message: InitMessage) {
  renderer = new WebGLRenderer(message.canvas);
  sourceWidth = message.width;
  sourceHeight = message.height;
  segmenter = new SegmentationManager();
  await segmenter.initialize(message.width, message.height);
  maskProcessor = new MaskProcessor();
  currentTuning = message.tuning;
  lastDebugAt = {};
  updateProcessingResolution(0, false);
  postDebug('init', 'Segmentation model ready', {
    sourceWidth,
    sourceHeight,
    processingWidth,
    processingHeight,
    model: 'selfie_segmenter.tflite'
  }, true);
  scheduleTick();
  postMessage({ type: 'ready' });
}

async function handleFrame(message: FrameMessage) {
  closeBitmap(pendingFrame);
  pendingFrame = message.frame;
  scheduleTick();
}

function handleDebugCanvases(message: DebugCanvasesMessage) {
  debugRawCanvas = message.rawCanvas;
  debugRawContext = null;
  debugCleanedCanvas = message.cleanedCanvas;
  debugCleanedContext = null;
}

async function processTick() {
  if (!renderer || !segmenter || !maskProcessor || !pendingFrame) {
    scheduleTick();
    return;
  }

  const frameStart = performance.now();
  const sourceFrame = pendingFrame;
  pendingFrame = null;

  const preprocessStart = performance.now();
  const processedBitmap = await drawForProcessing(sourceFrame);
  const { brightness, motion } = computeLuma(processedBitmap);
  const preprocessMs = performance.now() - preprocessStart;
  postDebug('preprocess', 'Camera frame prepared', {
    sourceWidth: sourceFrame.width,
    sourceHeight: sourceFrame.height,
    processingWidth: processedBitmap.width,
    processingHeight: processedBitmap.height,
    brightness: roundMetric(brightness),
    motion: roundMetric(motion),
    flipped: false,
    preprocessMs: roundMetric(preprocessMs)
  });

  const segmentationStart = performance.now();
  const segmentation = await segmenter.segment(processedBitmap, Math.round(frameStart));
  const segmentationMs = performance.now() - segmentationStart;
  const tuning = { ...currentTuning };
  const processedMask = maskProcessor.process(segmentation, currentTuning);
  const rawBranch = segmentation.branches[0];
  const rawMask = rawBranch?.confidenceMask ?? Float32Array.from(rawBranch?.categoryMask ?? new Uint8Array(processedMask.alphaMask.length), (value) => (value ?? 0) > 0 ? 1 : 0);
  const rawMaskSummary = summarizeMask(rawMask, 0.65);

  postDebug('segmentation', 'MediaPipe human segmentation finished', {
    branches: segmentation.branches.length,
    segmentationMs: roundMetric(segmentationMs),
    rawMean: roundMetric(rawMaskSummary.mean),
    rawCoverage: roundMetric(rawMaskSummary.coverage),
    rawMin: roundMetric(rawMaskSummary.min),
    rawMax: roundMetric(rawMaskSummary.max)
  });

  debugRawContext = drawMaskPreview(debugRawCanvas, debugRawContext, rawMask, processedBitmap.width, processedBitmap.height);
  debugCleanedContext = drawMaskPreview(debugCleanedCanvas, debugCleanedContext, processedMask.alphaMask, processedBitmap.width, processedBitmap.height);

  const cleanedSummary = summarizeMask(processedMask.alphaMask, 0.5);
  postDebug('mask', 'Foreground mask prepared', {
    alphaMean: roundMetric(cleanedSummary.mean),
    alphaCoverage: roundMetric(cleanedSummary.coverage),
    confidenceMean: roundMetric(processedMask.confidenceMean),
    foregroundRatio: roundMetric(processedMask.foregroundRatio),
    maskMotion: roundMetric(processedMask.motionMagnitude)
  });

  const combinedMotion = Math.max(motion, processedMask.motionMagnitude);
  const renderStart = performance.now();

  if ((processedMask.foregroundRatio < 0.01 || processedMask.foregroundRatio > 0.99) && performance.now() - lastMaskWarningAt > 3000) {
    lastMaskWarningAt = performance.now();
    console.warn(`Human mask looks suspicious (${(processedMask.foregroundRatio * 100).toFixed(1)}% coverage)`);
  }

  const renderArgs: RenderFrameArgs = {
    frame: processedBitmap,
    alphaMask: processedMask.alphaMask,
    confidenceMask: processedMask.confidenceMask,
    tuning
  };

  await renderer.renderFrame(renderArgs);
  const renderMs = performance.now() - renderStart;
  const latencyMs = performance.now() - frameStart;
  const fps = latencyMs > 0 ? 1000 / latencyMs : 0;

  postDebug('composite', 'Final composited frame rendered', {
    width: processedBitmap.width,
    height: processedBitmap.height,
    renderMs: roundMetric(renderMs),
    latencyMs: roundMetric(latencyMs),
    fps: roundMetric(fps),
    foregroundRatio: roundMetric(processedMask.foregroundRatio),
    maskMean: roundMetric(processedMask.maskMean)
  });

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

  const averagedStats = performanceTracker.snapshot();
  applyQualityFallback(fps, averagedStats.segmentationMs);

  postMessage({ type: 'stats', stats: averagedStats satisfies EngineStats });

  if (processedBitmap !== sourceFrame) processedBitmap.close();
  sourceFrame.close();
  postMessage({ type: 'frameProcessed' });
  scheduleTick();
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  if (message.type === 'init') { await handleInit(message); return; }
  if (message.type === 'frame') { await handleFrame(message); return; }
  if (message.type === 'tuning') { currentTuning = message.tuning; return; }
  if (message.type === 'background') {
    renderer?.setBackgroundBitmap(message.bitmap);
    postDebug('background', 'Background bitmap uploaded to renderer', {
      width: message.bitmap.width,
      height: message.bitmap.height
    }, true);
    return;
  }
  if (message.type === 'debugCanvases') {
    handleDebugCanvases(message);
    return;
  }
  if (message.type === 'resize') {
    sourceWidth = message.width;
    sourceHeight = message.height;
    updateProcessingResolution(qualityTierIndex, false);
    return;
  }
  if (message.type === 'stop') {
    if (tickHandle !== null) {
      if (typeof self.cancelAnimationFrame === 'function') self.cancelAnimationFrame(tickHandle);
      else self.clearTimeout(tickHandle);
      tickHandle = null;
    }
    closeBitmap(pendingFrame);
    pendingFrame = null;
    segmenter?.close();
    maskProcessor?.reset();
    renderer?.destroy();
    renderer = null;
    segmenter = null;
    maskProcessor = null;
    previousLuma = null;
    debugRawCanvas = null;
    debugRawContext = null;
    debugCleanedCanvas = null;
    debugCleanedContext = null;
  }
};
