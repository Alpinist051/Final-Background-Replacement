import {
  FilesetResolver,
  ImageSegmenter,
  type MPMask
} from '@mediapipe/tasks-vision';
import type { SegmentationFrameResult } from '@/types/engine';

type ImportFallbackHost = typeof globalThis & {
  import?: (specifier: string) => Promise<unknown>;
};

const importFallbackHost = globalThis as ImportFallbackHost;

if (typeof importFallbackHost.import !== 'function') {
  importFallbackHost.import = (specifier: string) => import(/* @vite-ignore */ specifier);
}

const VISION_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite';

function extractCategoryMask(mask: MPMask | undefined): Uint8Array | undefined {
  if (!mask) return undefined;
  if (mask.hasUint8Array()) {
    return mask.getAsUint8Array();
  }

  const source = mask.getAsFloat32Array();
  const output = new Uint8Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    output[i] = Math.max(0, Math.min(255, Math.round(source[i])));
  }
  return output;
}

function extractMaskFloats(mask: MPMask): Float32Array {
  return mask.hasFloat32Array()
    ? mask.getAsFloat32Array()
    : Float32Array.from(mask.getAsUint8Array(), (value) => value / 255);
}

function extractForegroundConfidenceMask(masks: MPMask[] | undefined): Float32Array | undefined {
  if (!masks?.length) return undefined;

  const floatMasks = masks.map(extractMaskFloats);
  const primary = floatMasks[0];
  const output = new Float32Array(primary.length);

  for (let i = 0; i < primary.length; i += 1) {
    let foregroundConfidence = 0;
    if (floatMasks.length > 1) {
      for (let maskIndex = 1; maskIndex < floatMasks.length; maskIndex += 1) {
        foregroundConfidence = Math.max(foregroundConfidence, floatMasks[maskIndex][i] ?? 0);
      }
    } else {
      foregroundConfidence = 1 - primary[i];
    }
    output[i] = Math.max(0, Math.min(1, foregroundConfidence));
  }

  return output;
}

function buildCategoryMaskFromConfidenceMasks(masks: MPMask[] | undefined): Uint8Array | undefined {
  if (!masks?.length) return undefined;

  const floatMasks = masks.map(extractMaskFloats);
  const length = floatMasks[0]?.length ?? 0;
  if (!length) return undefined;

  const output = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    let bestIndex = 0;
    let bestValue = floatMasks[0][i] ?? 0;
    for (let maskIndex = 1; maskIndex < floatMasks.length; maskIndex += 1) {
      const candidate = floatMasks[maskIndex][i] ?? 0;
      if (candidate > bestValue) {
        bestValue = candidate;
        bestIndex = maskIndex;
      }
    }
    output[i] = bestIndex;
  }

  return output;
}

function foregroundRatio(categoryMask: Uint8Array) {
  let count = 0;
  for (let i = 0; i < categoryMask.length; i += 1) {
    if (categoryMask[i] !== 0) count += 1;
  }
  return count / Math.max(1, categoryMask.length);
}

export class SegmentationManager {
  private segmenter: ImageSegmenter | null = null;
  private labels: string[] = [];

  async initialize(retry = 0): Promise<void> {
    if (this.segmenter) return;

    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);

    try {
      this.segmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: 'GPU' as const
        },
        runningMode: 'VIDEO',
        outputCategoryMask: true,
        outputConfidenceMasks: true
      });
    } catch (err) {
      if (retry < 1) {
        // One-time graceful fallback to CPU
        console.warn('GPU delegate failed, falling back to CPU (still high quality)');
        this.segmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' as const },
          runningMode: 'VIDEO',
          outputCategoryMask: true,
          outputConfidenceMasks: true
        });
      } else {
        throw err;
      }
    }

    this.labels = this.segmenter.getLabels();
  }

  async segment(frame: ImageBitmap, timestampMs: number): Promise<SegmentationFrameResult> {
    if (!this.segmenter) await this.initialize();

    const result = this.segmenter!.segmentForVideo(frame, timestampMs);

    let categoryMask = extractCategoryMask(result.categoryMask);
    if (categoryMask && foregroundRatio(categoryMask) < 0.001) {
      const derived = buildCategoryMaskFromConfidenceMasks(result.confidenceMasks);
      if (derived) {
        categoryMask = derived;
      }
    }
    const confidenceMaskRaw = extractForegroundConfidenceMask(result.confidenceMasks);

    if (!categoryMask || !(categoryMask instanceof Uint8Array)) {
      throw new Error('MediaPipe failed to return category mask');
    }

    if (this.labels.length && categoryMask && foregroundRatio(categoryMask) < 0.001) {
      console.warn('MediaPipe category mask stayed empty after fallback. Labels:', this.labels);
    }

    const output: SegmentationFrameResult = {
      width: frame.width,   // force exact input resolution (super-perfect alignment)
      height: frame.height,
      categoryMask,
      confidenceMask: confidenceMaskRaw instanceof Float32Array ? confidenceMaskRaw : undefined,
      labels: this.labels
    };

    // Clean up MediaPipe internal masks immediately
    result.categoryMask?.close();
    result.confidenceMasks?.forEach(m => m.close());

    return output;
  }

  close() {
    this.segmenter?.close();
    this.segmenter = null;
    this.labels = [];
  }
}
