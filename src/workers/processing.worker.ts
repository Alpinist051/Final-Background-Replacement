import { createAnalysisCanvas } from '@/utils/canvasUtils';
import { createPerformanceTracker } from '@/utils/performance';
import type { BackgroundSource, EngineStats, VirtualBackgroundTuning } from '@/types/engine';
import { SegmentationManager } from '@/core/SegmentationManager';
import { MaskProcessor, type ProcessedMask } from '@/core/MaskProcessor';
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
let subjectCanvas: OffscreenCanvas | null = null;
let subjectContext: OffscreenCanvasRenderingContext2D | null = null;
let subjectFramesSinceRefresh = Number.POSITIVE_INFINITY;

type QualityTier = {
  maxWidth: number;
  maxHeight: number;
  temporalAlpha: number;
};

const QUALITY_TIERS: QualityTier[] = [
  { maxWidth: 640, maxHeight: 480, temporalAlpha: 0.76 },
  { maxWidth: 512, maxHeight: 384, temporalAlpha: 0.82 },
  { maxWidth: 384, maxHeight: 288, temporalAlpha: 0.9 }
];

let currentTuning: VirtualBackgroundTuning = {
  temporalAlpha: 0.76,
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
let sourceWidth = 1280;
let sourceHeight = 720;
let processingWidth = 640;
let processingHeight = 480;
let qualityTierIndex = 0;
let lowFpsStreak = 0;
let highFpsStreak = 0;
let tickHandle: number | null = null;
let lastMaskWarningAt = 0;
const TARGET_FPS = 30;
const LOW_FPS_THRESHOLD = 25;
const LOW_FPS_FRAMES_BEFORE_DROP = 4;
const HIGH_FPS_FRAMES_BEFORE_RAISE = 45;

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
  currentTuning = { ...currentTuning, temporalAlpha: tier.temporalAlpha };
  renderer?.resize(processingWidth, processingHeight);
  segmenter?.resetCache();
  maskProcessor?.reset();
  subjectFramesSinceRefresh = Number.POSITIVE_INFINITY;
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

function ensureSubjectCanvas(width: number, height: number) {
  if (!subjectCanvas || subjectCanvas.width !== width || subjectCanvas.height !== height) {
    subjectCanvas = new OffscreenCanvas(width, height);
    subjectContext = subjectCanvas.getContext('2d', { willReadFrequently: true });
  }
}

async function drawForProcessing(bitmap: ImageBitmap) {
  ensureProcessingCanvas();
  if (!processingCanvas || !processingContext) return bitmap;
  processingContext.clearRect(0, 0, processingCanvas.width, processingCanvas.height);
  processingContext.setTransform(1, 0, 0, -1, 0, processingCanvas.height);
  processingContext.drawImage(bitmap, 0, 0, processingCanvas.width, processingCanvas.height);
  processingContext.setTransform(1, 0, 0, 1, 0, 0);
  return createImageBitmap(processingCanvas);
}

function chooseSubjectProcessingBounds(qualityTier: number, fps: number, segmentationMs: number, motion: number) {
  if (qualityTier >= 2) {
    return fps < 20 || segmentationMs > 80
      ? { maxWidth: 256, maxHeight: 256 }
      : { maxWidth: 288, maxHeight: 288 };
  }

  if (fps > 28 && segmentationMs < 55) {
    return { maxWidth: 384, maxHeight: 384 };
  }

  if (fps < 24 || segmentationMs > 60) {
    return { maxWidth: 288, maxHeight: 288 };
  }

  if (motion > 0.08) {
    return { maxWidth: 384, maxHeight: 384 };
  }

  return { maxWidth: 352, maxHeight: 352 };
}

async function drawForSubjectProcessing(bitmap: ImageBitmap, maxWidth: number, maxHeight: number) {
  const fit = fitWithinBounds(bitmap.width, bitmap.height, maxWidth, maxHeight);
  ensureSubjectCanvas(fit.width, fit.height);
  if (!subjectCanvas || !subjectContext) return null;

  subjectContext.clearRect(0, 0, subjectCanvas.width, subjectCanvas.height);
  subjectContext.setTransform(1, 0, 0, -1, 0, subjectCanvas.height);
  subjectContext.drawImage(bitmap, 0, 0, subjectCanvas.width, subjectCanvas.height);
  return createImageBitmap(subjectCanvas);
}

function chooseSubjectRefreshInterval(motion: number, fps: number, segmentationMs: number, qualityTier: number) {
  const overloaded = fps < 24 || segmentationMs > 60 || qualityTier >= 2;

  if (motion > 0.05) {
    return overloaded ? 2 : 1;
  }

  if (motion > 0.025) {
    return overloaded ? 2 : 1;
  }

  if (overloaded) {
    return qualityTier >= 2 ? 4 : 3;
  }

  return 2;
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

function boostTuning(brightness: number, motion: number, processedMask: ProcessedMask) {
  const boosted = { ...currentTuning };
  const motionSignal = Math.max(motion, processedMask.motionMagnitude);
  const motionIntensity = Math.min(1, Math.max(0, motionSignal * 4.5));

  if (motionIntensity > 0.7 || processedMask.motionMagnitude > 0.2) {
    boosted.confidenceBoost = Math.max(boosted.confidenceBoost, 1.45);
  }
  if (brightness < 80) {
    boosted.confidenceBoost = Math.min(2.5, boosted.confidenceBoost * boosted.brightnessBoost * 1.25);
  }
  boosted.confidenceBoost = Math.min(2.7, boosted.confidenceBoost);
  boosted.temporalAlpha = Math.max(0.64, Math.min(0.9, boosted.temporalAlpha - motionIntensity * 0.12));
  boosted.motionBoost = Math.max(0.25, Math.min(1, 0.28 + motionIntensity * 0.72));
  return boosted;
}

function applyQualityFallback(fps: number, segmentationMs: number) {
  if (fps < LOW_FPS_THRESHOLD || segmentationMs > 70) {
    lowFpsStreak += 1;
    highFpsStreak = 0;
  } else if (fps > 33 && segmentationMs < 45) {
    highFpsStreak += 1;
    lowFpsStreak = 0;
  } else {
    lowFpsStreak = 0;
    highFpsStreak = 0;
  }

  if (lowFpsStreak >= LOW_FPS_FRAMES_BEFORE_DROP && qualityTierIndex < QUALITY_TIERS.length - 1) {
    updateProcessingResolution(qualityTierIndex + 1, true);
    lowFpsStreak = 0;
    highFpsStreak = 0;
  } else if (highFpsStreak >= HIGH_FPS_FRAMES_BEFORE_RAISE && qualityTierIndex > 0) {
    updateProcessingResolution(qualityTierIndex - 1, true);
    lowFpsStreak = 0;
    highFpsStreak = 0;
  }
}

async function handleInit(message: InitMessage) {
  renderer = new WebGLRenderer(message.canvas);
  sourceWidth = message.width;
  sourceHeight = message.height;
  renderer.setBackground(message.background);
  segmenter = new SegmentationManager();
  await segmenter.initialize(message.width, message.height);
  maskProcessor = new MaskProcessor();
  currentTuning = message.tuning;
  currentBackground = message.background;
  subjectFramesSinceRefresh = Number.POSITIVE_INFINITY;
  updateProcessingResolution(0, false);
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
  const recentStats = performanceTracker.snapshot();
  const subjectRefreshInterval = chooseSubjectRefreshInterval(motion, recentStats.fps || TARGET_FPS, recentStats.segmentationMs, qualityTierIndex);
  const subjectBounds = chooseSubjectProcessingBounds(qualityTierIndex, recentStats.fps || TARGET_FPS, recentStats.segmentationMs, motion);
  const refreshSubject = subjectFramesSinceRefresh === Number.POSITIVE_INFINITY || subjectFramesSinceRefresh >= subjectRefreshInterval;

  let subjectBitmap: ImageBitmap | null = null;
  if (refreshSubject) {
    // Use the original frame here so the subject branch is transformed exactly once.
    // Feeding it the already-processed bitmap would introduce a second flip and make
    // the subject mask drift away from the selfie branch.
    subjectBitmap = await drawForSubjectProcessing(sourceFrame, subjectBounds.maxWidth, subjectBounds.maxHeight);
  }

  const segmentationStart = performance.now();
  let segmentation;
  try {
    segmentation = await segmenter.segment(processedBitmap, Math.round(frameStart), {
      refreshSubject,
      subjectFrame: subjectBitmap ?? undefined
    });
  } finally {
    subjectBitmap?.close();
  }
  const segmentationMs = performance.now() - segmentationStart;
  const processedMask = maskProcessor.process(segmentation, currentTuning, motion);
  subjectFramesSinceRefresh = refreshSubject ? 0 : subjectFramesSinceRefresh + 1;

  const tuning = boostTuning(brightness, motion, processedMask);

  const combinedMotion = Math.max(motion, processedMask.motionMagnitude);
  const renderStart = performance.now();

  if ((processedMask.foregroundRatio < 0.01 || processedMask.foregroundRatio > 0.99) && performance.now() - lastMaskWarningAt > 3000) {
    lastMaskWarningAt = performance.now();
    console.warn(`Foreground mask looks suspicious (${(processedMask.foregroundRatio * 100).toFixed(1)}% coverage)`);
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

  const averagedStats = performanceTracker.snapshot();
  applyQualityFallback(fps, averagedStats.segmentationMs);

  postMessage({ type: 'stats', stats: averagedStats satisfies EngineStats });

  if (processedBitmap !== sourceFrame) processedBitmap.close();
  sourceFrame.close();
  sourceBackgroundFrame?.close?.();
  postMessage({ type: 'frameProcessed' });
  scheduleTick();
}

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  if (message.type === 'init') { await handleInit(message); return; }
  if (message.type === 'frame') { await handleFrame(message); return; }
  if (message.type === 'tuning') { currentTuning = message.tuning; return; }
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
    sourceWidth = message.width;
    sourceHeight = message.height;
    segmenter?.resetCache();
    subjectFramesSinceRefresh = Number.POSITIVE_INFINITY;
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
    subjectFramesSinceRefresh = Number.POSITIVE_INFINITY;
    subjectCanvas = null;
    subjectContext = null;
  }
};
