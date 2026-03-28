import {
  FilesetResolver,
  ImageSegmenter,
  type MPMask
} from '@mediapipe/tasks-vision';
import type {
  SegmentationBranchResult,
  SegmentationFrameResult
} from '@/types/engine';

type ImportFallbackHost = typeof globalThis & {
  import?: (specifier: string) => Promise<unknown>;
};

const importFallbackHost = globalThis as ImportFallbackHost;

if (typeof importFallbackHost.import !== 'function') {
  importFallbackHost.import = (specifier: string) => import(/* @vite-ignore */ specifier);
}

const VISION_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';
const SELFIE_MODEL_SQUARE_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite';
const SELFIE_MODEL_LANDSCAPE_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/1/selfie_segmenter_landscape.tflite';

type SegmenterSlot = {
  segmenter: ImageSegmenter;
  labels: string[];
};

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

function findBackgroundIndex(labels: string[]) {
  return labels.findIndex((label) => /background/i.test(label));
}

function extractForegroundConfidenceMask(
  masks: MPMask[] | undefined,
  labels: string[],
  categoryMask?: Uint8Array
): Float32Array | undefined {
  if (!masks?.length) return undefined;

  const floatMasks = masks.map(extractMaskFloats);
  const primary = floatMasks[0];
  const backgroundIndex = findBackgroundIndex(labels);
  const output = new Float32Array(primary.length);

  for (let i = 0; i < primary.length; i += 1) {
    let foregroundConfidence = 0;
    if (categoryMask) {
      const selectedIndex = categoryMask[i] ?? 0;
      const label = labels[selectedIndex] ?? '';
      if (!/background/i.test(label)) {
        foregroundConfidence = floatMasks[selectedIndex]?.[i] ?? 0;
      } else if (backgroundIndex >= 0) {
        foregroundConfidence = 1 - (floatMasks[backgroundIndex][i] ?? 0);
      }
    }

    if (!foregroundConfidence) {
      if (floatMasks.length === 1) {
        const singleLabel = labels[0] ?? '';
        foregroundConfidence = /background/i.test(singleLabel)
          ? 1 - primary[i]
          : primary[i];
      } else if (backgroundIndex >= 0) {
        foregroundConfidence = 1 - (floatMasks[backgroundIndex][i] ?? 0);
      } else {
        for (let maskIndex = 0; maskIndex < floatMasks.length; maskIndex += 1) {
          const label = labels[maskIndex] ?? '';
          if (/background/i.test(label)) continue;
          foregroundConfidence = Math.max(foregroundConfidence, floatMasks[maskIndex][i] ?? 0);
        }
      }
    }

    output[i] = Math.max(0, Math.min(1, foregroundConfidence));
  }

  return output;
}

function buildCategoryMaskFromConfidenceMasks(masks: MPMask[] | undefined, labels: string[]): Uint8Array | undefined {
  if (!masks?.length) return undefined;

  const floatMasks = masks.map(extractMaskFloats);
  const length = floatMasks[0]?.length ?? 0;
  if (!length) return undefined;
  const backgroundIndex = findBackgroundIndex(labels);

  const output = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    let bestIndex = backgroundIndex >= 0 ? backgroundIndex : 0;
    let bestValue = -Infinity;
    for (let maskIndex = 0; maskIndex < floatMasks.length; maskIndex += 1) {
      const candidate = floatMasks[maskIndex][i] ?? 0;
      if (candidate > bestValue) {
        bestValue = candidate;
        bestIndex = maskIndex;
      }
    }
    output[i] = bestValue >= 0.12 ? bestIndex : (backgroundIndex >= 0 ? backgroundIndex : 0);
  }

  return output;
}

function foregroundRatio(categoryMask: Uint8Array, labels: string[]) {
  let count = 0;
  for (let i = 0; i < categoryMask.length; i += 1) {
    const label = labels[categoryMask[i] ?? 0] ?? '';
    if (label && !/background/i.test(label)) count += 1;
  }
  return count / Math.max(1, categoryMask.length);
}

function resampleCategoryMask(source: Uint8Array, sourceWidth: number, sourceHeight: number, targetWidth: number, targetHeight: number) {
  if (sourceWidth === targetWidth && sourceHeight === targetHeight) return source;

  const output = new Uint8Array(targetWidth * targetHeight);
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor((y + 0.5) * sourceHeight / targetHeight));
    const sourceRow = sourceY * sourceWidth;
    const targetRow = y * targetWidth;
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = Math.min(sourceWidth - 1, Math.floor((x + 0.5) * sourceWidth / targetWidth));
      output[targetRow + x] = source[sourceRow + sourceX] ?? 0;
    }
  }
  return output;
}

function resampleConfidenceMask(source: Float32Array, sourceWidth: number, sourceHeight: number, targetWidth: number, targetHeight: number) {
  if (sourceWidth === targetWidth && sourceHeight === targetHeight) return source;

  const output = new Float32Array(targetWidth * targetHeight);
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor((y + 0.5) * sourceHeight / targetHeight));
    const sourceRow = sourceY * sourceWidth;
    const targetRow = y * targetWidth;
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = Math.min(sourceWidth - 1, Math.floor((x + 0.5) * sourceWidth / targetWidth));
      output[targetRow + x] = source[sourceRow + sourceX] ?? 0;
    }
  }
  return output;
}

function chooseSelfieModelCandidates(width: number, height: number) {
  const primaryFallback = width >= height ? SELFIE_MODEL_LANDSCAPE_URL : SELFIE_MODEL_SQUARE_URL;
  const secondaryFallback = width >= height ? SELFIE_MODEL_SQUARE_URL : SELFIE_MODEL_LANDSCAPE_URL;
  return [primaryFallback, secondaryFallback];
}

async function createSegmenter(vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>, modelAssetPath: string) {
  try {
    return await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: 'GPU' as const
      },
      runningMode: 'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: true
    });
  } catch (err) {
    console.warn('GPU delegate failed for segmentation model, falling back to CPU.', err);
    return ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: 'CPU' as const
      },
      runningMode: 'VIDEO',
      outputCategoryMask: true,
      outputConfidenceMasks: true
    });
  }
}

function createBranchResult(
  width: number,
  height: number,
  categoryMask: Uint8Array,
  confidenceMask: Float32Array | undefined,
  labels: string[],
  ageMs: number
): SegmentationBranchResult {
  return {
    kind: 'selfie',
    width,
    height,
    categoryMask,
    confidenceMask,
    labels,
    ageMs
  };
}

export class SegmentationManager {
  private selfieSegmenter: SegmenterSlot | null = null;

  async initialize(sourceWidth = 1280, sourceHeight = 720): Promise<void> {
    if (this.selfieSegmenter) return;

    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);

    const selfieModelCandidates = chooseSelfieModelCandidates(sourceWidth, sourceHeight);

    if (!this.selfieSegmenter) {
      for (const modelAssetPath of selfieModelCandidates) {
        try {
          const selfieSegmenter = await createSegmenter(vision, modelAssetPath);
          this.selfieSegmenter = {
            segmenter: selfieSegmenter,
            labels: selfieSegmenter.getLabels()
          };
          break;
        } catch (error) {
          console.warn('Selfie segmentation model failed to initialize.', error);
        }
      }
    }

    if (!this.selfieSegmenter) {
      throw new Error('Unable to initialize the selfie segmentation model.');
    }
  }

  private segmentBranch(
    slot: SegmenterSlot,
    frame: ImageBitmap,
    timestampMs: number,
    ageMs = 0,
    outputWidth = frame.width,
    outputHeight = frame.height
  ): SegmentationBranchResult {
    const result = slot.segmenter.segmentForVideo(frame, timestampMs);
    const sourceWidth = frame.width;
    const sourceHeight = frame.height;

    let categoryMask = extractCategoryMask(result.categoryMask);
    if (!categoryMask || foregroundRatio(categoryMask, slot.labels) < 0.001) {
      const derivedMask = buildCategoryMaskFromConfidenceMasks(result.confidenceMasks, slot.labels);
      if (derivedMask) {
        categoryMask = derivedMask;
      }
    }
    const confidenceMask = extractForegroundConfidenceMask(result.confidenceMasks, slot.labels, categoryMask);

    if (!categoryMask || !(categoryMask instanceof Uint8Array)) {
      throw new Error('MediaPipe failed to return a category mask.');
    }

    const resizedCategoryMask = resampleCategoryMask(categoryMask, sourceWidth, sourceHeight, outputWidth, outputHeight);
    const resizedConfidenceMask = confidenceMask
      ? resampleConfidenceMask(confidenceMask, sourceWidth, sourceHeight, outputWidth, outputHeight)
      : undefined;

    const branch = createBranchResult(outputWidth, outputHeight, resizedCategoryMask, resizedConfidenceMask, slot.labels, ageMs);

    result.categoryMask?.close();
    result.confidenceMasks?.forEach((mask) => mask.close());

    return branch;
  }

  async segment(frame: ImageBitmap, timestampMs: number): Promise<SegmentationFrameResult> {
    if (!this.selfieSegmenter) {
      await this.initialize(frame.width, frame.height);
    }

    if (!this.selfieSegmenter) {
      throw new Error('Segmentation model is not initialized.');
    }

    const branches: SegmentationBranchResult[] = [];

    if (this.selfieSegmenter) {
      try {
        branches.push(this.segmentBranch(this.selfieSegmenter, frame, timestampMs, 0));
      } catch (error) {
        console.warn('Selfie segmentation frame failed.', error);
      }
    }

    if (!branches.length) {
      throw new Error('No segmentation branches are available.');
    }

    return {
      width: frame.width,
      height: frame.height,
      branches
    };
  }

  close() {
    this.selfieSegmenter?.segmenter.close();
    this.selfieSegmenter = null;
  }
}
