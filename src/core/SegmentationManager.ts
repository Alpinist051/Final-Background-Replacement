import {
  FilesetResolver,
  ImageSegmenter,
  type MPMask
} from '@mediapipe/tasks-vision';
import type {
  SegmentationBranchResult,
  SegmentationFrameResult
} from '@/types/engine';
import {
  getBackgroundIndex,
  getSemanticLabelPriority
} from './segmentationLabels';

type ImportFallbackHost = typeof globalThis & {
  import?: (specifier: string) => Promise<unknown>;
};

const importFallbackHost = globalThis as ImportFallbackHost;

if (typeof importFallbackHost.import !== 'function') {
  importFallbackHost.import = (specifier: string) => import(/* @vite-ignore */ specifier);
}

const VISION_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';
const MULTICLASS_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite';

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

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

function extractForegroundConfidenceMask(
  masks: MPMask[] | undefined,
  labels: string[],
  categoryMask?: Uint8Array
): Float32Array | undefined {
  if (!masks?.length) return undefined;

  const floatMasks = masks.map(extractMaskFloats);
  const primary = floatMasks[0];
  const backgroundIndex = getBackgroundIndex(labels);
  const output = new Float32Array(primary.length);

  for (let i = 0; i < primary.length; i += 1) {
    const backgroundConfidence = backgroundIndex >= 0 ? (floatMasks[backgroundIndex][i] ?? 0) : 0;
    let foregroundConfidence = backgroundIndex >= 0 ? 1 - backgroundConfidence : 0;

    if (categoryMask) {
      const selectedIndex = categoryMask[i] ?? 0;
      const label = labels[selectedIndex] ?? '';
      if (selectedIndex !== backgroundIndex && label && !/background/i.test(label)) {
        foregroundConfidence = Math.max(
          foregroundConfidence,
          (floatMasks[selectedIndex]?.[i] ?? 0) * getSemanticLabelPriority(label)
        );
      }
    }

    for (let maskIndex = 0; maskIndex < floatMasks.length; maskIndex += 1) {
      const label = labels[maskIndex] ?? '';
      if (!label || /background/i.test(label)) continue;
      const candidate = (floatMasks[maskIndex][i] ?? 0) * getSemanticLabelPriority(label);
      foregroundConfidence = Math.max(foregroundConfidence, candidate);
    }

    if (!foregroundConfidence) {
      if (floatMasks.length === 1) {
        const singleLabel = labels[0] ?? '';
        foregroundConfidence = /background/i.test(singleLabel) ? 1 - primary[i] : primary[i];
      } else if (backgroundIndex >= 0) {
        foregroundConfidence = 1 - (floatMasks[backgroundIndex][i] ?? 0);
      }
    }

    output[i] = clamp01(foregroundConfidence);
  }

  return output;
}

function buildCategoryMaskFromConfidenceMasks(masks: MPMask[] | undefined, labels: string[]): Uint8Array | undefined {
  if (!masks?.length) return undefined;

  const floatMasks = masks.map(extractMaskFloats);
  const length = floatMasks[0]?.length ?? 0;
  if (!length) return undefined;
  const backgroundIndex = getBackgroundIndex(labels);

  const output = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    let bestIndex = backgroundIndex >= 0 ? backgroundIndex : 0;
    let bestValue = -Infinity;
    let bestForegroundIndex = bestIndex;
    let bestForegroundValue = -Infinity;
    const backgroundValue = backgroundIndex >= 0 ? (floatMasks[backgroundIndex][i] ?? 0) : 0;
    const backgroundScore = backgroundValue * 0.92;

    for (let maskIndex = 0; maskIndex < floatMasks.length; maskIndex += 1) {
      const candidate = floatMasks[maskIndex][i] ?? 0;
      const label = labels[maskIndex] ?? '';
      const priority = getSemanticLabelPriority(label);
      const score = candidate * priority;
      if (score > bestValue) {
        bestValue = score;
        bestIndex = maskIndex;
      }
      if (maskIndex !== backgroundIndex && score > bestForegroundValue) {
        bestForegroundValue = score;
        bestForegroundIndex = maskIndex;
      }
    }

    const foregroundWins = bestForegroundIndex !== backgroundIndex && bestForegroundValue >= Math.max(0.08, backgroundScore - 0.02);
    output[i] = foregroundWins ? bestForegroundIndex : bestIndex;
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

function chooseMultiPersonModelCandidates() {
  return [MULTICLASS_MODEL_URL];
}

async function createSegmenter(vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>, modelAssetPath: string) {
  try {
    return await ImageSegmenter.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: 'GPU' as const
      },
      runningMode: 'VIDEO',
      displayNamesLocale: 'en',
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
      displayNamesLocale: 'en',
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
    kind: 'human',
    width,
    height,
    categoryMask,
    confidenceMask,
    labels,
    ageMs
  };
}

export class SegmentationManager {
  private humanSegmenter: SegmenterSlot | null = null;

  async initialize(_sourceWidth = 1280, _sourceHeight = 720): Promise<void> {
    if (this.humanSegmenter) return;

    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);

    const humanModelCandidates = chooseMultiPersonModelCandidates();

    if (!this.humanSegmenter) {
      for (const modelAssetPath of humanModelCandidates) {
        try {
          const humanSegmenter = await createSegmenter(vision, modelAssetPath);
          this.humanSegmenter = {
            segmenter: humanSegmenter,
            labels: humanSegmenter.getLabels()
          };
          break;
        } catch (error) {
          console.warn('Multi-person segmentation model failed to initialize.', error);
        }
      }
    }

    if (!this.humanSegmenter) {
      throw new Error('Unable to initialize the multi-person segmentation model.');
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
    if (!this.humanSegmenter) {
      await this.initialize(frame.width, frame.height);
    }

    if (!this.humanSegmenter) {
      throw new Error('Segmentation model is not initialized.');
    }

    const branches: SegmentationBranchResult[] = [];

    if (this.humanSegmenter) {
      try {
        branches.push(this.segmentBranch(this.humanSegmenter, frame, timestampMs, 0));
      } catch (error) {
        console.warn('Multi-person segmentation frame failed.', error);
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
    this.humanSegmenter?.segmenter.close();
    this.humanSegmenter = null;
  }
}
