import {
  FilesetResolver,
  ImageSegmenter,
  type MPMask
} from '@mediapipe/tasks-vision';
import type {
  SegmentationBranchKind,
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
const SUBJECT_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite';

type SegmenterSlot = {
  segmenter: ImageSegmenter;
  kind: SegmentationBranchKind;
  labels: string[];
};

type CachedBranch = SegmentationBranchResult & {
  updatedAtMs: number;
};

export type SegmentOptions = {
  refreshSubject?: boolean;
  subjectFrame?: ImageBitmap;
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

function buildCategoryMaskFromConfidenceMasks(masks: MPMask[] | undefined, labels: string[]): Uint8Array | undefined {
  if (!masks?.length) return undefined;

  const floatMasks = masks.map(extractMaskFloats);
  const length = floatMasks[0]?.length ?? 0;
  if (!length) return undefined;

  const classWeights = new Float32Array(Math.max(1, labels.length));
  for (let i = 0; i < classWeights.length; i += 1) {
    const label = labels[i]?.toLowerCase?.() ?? '';
    if (/background/i.test(label)) {
      classWeights[i] = 1;
    } else if (/(person|human|selfie)/i.test(label)) {
      classWeights[i] = 1.08;
    } else if (/(cat|dog|horse|cow|sheep|bird|car|bus|train|bicycle|motorbike|boat|chair|sofa|table|plant|potted plant)/i.test(label)) {
      classWeights[i] = 1.04;
    } else {
      classWeights[i] = 1.0;
    }
  }

  const output = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    const backgroundValue = floatMasks[0][i] ?? 0;
    let bestIndex = 0;
    let bestValue = backgroundValue * classWeights[0];
    for (let maskIndex = 1; maskIndex < floatMasks.length; maskIndex += 1) {
      const candidate = (floatMasks[maskIndex][i] ?? 0) * (classWeights[maskIndex] ?? 1);
      if (candidate > bestValue) {
        bestValue = candidate;
        bestIndex = maskIndex;
      }
    }
    output[i] = bestIndex !== 0 && bestValue >= 0.12 && bestValue >= backgroundValue * 0.74 ? bestIndex : 0;
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

function chooseSelfieModelUrl(width: number, height: number) {
  return width >= height ? SELFIE_MODEL_LANDSCAPE_URL : SELFIE_MODEL_SQUARE_URL;
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
  kind: SegmentationBranchKind,
  width: number,
  height: number,
  categoryMask: Uint8Array,
  confidenceMask: Float32Array | undefined,
  labels: string[],
  ageMs: number
): SegmentationBranchResult {
  return {
    kind,
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
  private subjectSegmenter: SegmenterSlot | null = null;
  private selfieModelUrl: string = SELFIE_MODEL_LANDSCAPE_URL;
  private cachedSubjectBranch: CachedBranch | null = null;

  async initialize(sourceWidth = 1280, sourceHeight = 720): Promise<void> {
    if (this.selfieSegmenter && this.subjectSegmenter) return;

    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);

    this.selfieModelUrl = chooseSelfieModelUrl(sourceWidth, sourceHeight);

    if (!this.selfieSegmenter) {
      try {
        const selfieSegmenter = await createSegmenter(vision, this.selfieModelUrl);
        this.selfieSegmenter = {
          segmenter: selfieSegmenter,
          kind: 'selfie',
          labels: selfieSegmenter.getLabels()
        };
      } catch (error) {
        console.warn('Selfie segmentation model failed to initialize.', error);
      }
    }

    if (!this.subjectSegmenter) {
      try {
        const subjectSegmenter = await createSegmenter(vision, SUBJECT_MODEL_URL);
        this.subjectSegmenter = {
          segmenter: subjectSegmenter,
          kind: 'subject',
          labels: subjectSegmenter.getLabels()
        };
      } catch (error) {
        console.warn('Subject segmentation model failed to initialize. Falling back to selfie-only mode.', error);
      }
    }

    if (!this.selfieSegmenter && !this.subjectSegmenter) {
      throw new Error('Unable to initialize any segmentation model.');
    }

    this.cachedSubjectBranch = null;
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
    const derivedMask = buildCategoryMaskFromConfidenceMasks(result.confidenceMasks, slot.labels);
    if (derivedMask && (!categoryMask || foregroundRatio(categoryMask) < 0.001)) {
      categoryMask = derivedMask;
    }
    const confidenceMask = extractForegroundConfidenceMask(result.confidenceMasks);

    if (!categoryMask || !(categoryMask instanceof Uint8Array)) {
      throw new Error('MediaPipe failed to return a category mask.');
    }

    const resizedCategoryMask = resampleCategoryMask(categoryMask, sourceWidth, sourceHeight, outputWidth, outputHeight);
    const resizedConfidenceMask = confidenceMask
      ? resampleConfidenceMask(confidenceMask, sourceWidth, sourceHeight, outputWidth, outputHeight)
      : undefined;

    const branch = createBranchResult(slot.kind, outputWidth, outputHeight, resizedCategoryMask, resizedConfidenceMask, slot.labels, ageMs);

    result.categoryMask?.close();
    result.confidenceMasks?.forEach((mask) => mask.close());

    return branch;
  }

  async segment(frame: ImageBitmap, timestampMs: number, options: SegmentOptions = {}): Promise<SegmentationFrameResult> {
    if (!this.selfieSegmenter && !this.subjectSegmenter) {
      await this.initialize(frame.width, frame.height);
    }

    if (!this.selfieSegmenter && !this.subjectSegmenter) {
      throw new Error('Segmentation models are not initialized.');
    }

    const branches: SegmentationBranchResult[] = [];

    if (this.selfieSegmenter) {
      try {
        branches.push(this.segmentBranch(this.selfieSegmenter, frame, timestampMs, 0));
      } catch (error) {
        console.warn('Selfie segmentation frame failed.', error);
      }
    }

    let subjectBranch: CachedBranch | null = this.cachedSubjectBranch;
    if (this.subjectSegmenter && options.refreshSubject) {
      const subjectSource = options.subjectFrame ?? frame;
      try {
        const nextSubject = this.segmentBranch(this.subjectSegmenter, subjectSource, timestampMs, 0, frame.width, frame.height);
        subjectBranch = {
          ...nextSubject,
          updatedAtMs: timestampMs
        };
        this.cachedSubjectBranch = subjectBranch;
      } catch (error) {
        console.warn('Subject segmentation frame failed. Reusing the cached subject branch when available.', error);
      }
    }

    if (subjectBranch) {
      branches.push({
        ...subjectBranch,
        ageMs: Math.max(0, timestampMs - subjectBranch.updatedAtMs)
      });
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
    this.subjectSegmenter?.segmenter.close();
    this.selfieSegmenter = null;
    this.subjectSegmenter = null;
    this.cachedSubjectBranch = null;
  }

  resetCache() {
    this.cachedSubjectBranch = null;
  }
}
