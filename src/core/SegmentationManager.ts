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
const SELFIE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';
const DEFAULT_LABELS = ['background', 'person'];

type SelfieSegmenterSlot = {
  segmenter: ImageSegmenter;
  labels: string[];
};

function normalizeLabels(labels: readonly string[]) {
  return labels.length > 0 ? [...labels] : [...DEFAULT_LABELS];
}

function extractBinaryMask(mask: MPMask | undefined, floatThreshold = 0.5): Uint8Array | undefined {
  if (!mask) return undefined;

  if (mask.hasUint8Array()) {
    const source = mask.getAsUint8Array();
    const output = new Uint8Array(source.length);
    for (let i = 0; i < source.length; i += 1) {
      output[i] = (source[i] ?? 0) > 0 ? 1 : 0;
    }
    return output;
  }

  const source = mask.getAsFloat32Array();
  const output = new Uint8Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    output[i] = (source[i] ?? 0) >= floatThreshold ? 1 : 0;
  }
  return output;
}

function extractMaskFloats(mask: MPMask): Float32Array {
  return mask.hasFloat32Array()
    ? mask.getAsFloat32Array()
    : Float32Array.from(mask.getAsUint8Array(), (value) => value / 255);
}

function resampleFloatMask(
  source: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number
) {
  if (sourceWidth === targetWidth && sourceHeight === targetHeight) return source;

  const output = new Float32Array(targetWidth * targetHeight);
  const maxSourceX = sourceWidth - 1;
  const maxSourceY = sourceHeight - 1;
  for (let y = 0; y < targetHeight; y += 1) {
    const sourceY = Math.max(0, Math.min(maxSourceY, ((y + 0.5) * sourceHeight / targetHeight) - 0.5));
    const y0 = Math.floor(sourceY);
    const y1 = Math.min(maxSourceY, y0 + 1);
    const yLerp = sourceY - y0;
    const targetRow = y * targetWidth;
    for (let x = 0; x < targetWidth; x += 1) {
      const sourceX = Math.max(0, Math.min(maxSourceX, ((x + 0.5) * sourceWidth / targetWidth) - 0.5));
      const x0 = Math.floor(sourceX);
      const x1 = Math.min(maxSourceX, x0 + 1);
      const xLerp = sourceX - x0;

      const topLeft = source[y0 * sourceWidth + x0] ?? 0;
      const topRight = source[y0 * sourceWidth + x1] ?? 0;
      const bottomLeft = source[y1 * sourceWidth + x0] ?? 0;
      const bottomRight = source[y1 * sourceWidth + x1] ?? 0;

      const top = topLeft + (topRight - topLeft) * xLerp;
      const bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
      output[targetRow + x] = top + (bottom - top) * yLerp;
    }
  }
  return output;
}

function chooseSelfieModelCandidates() {
  return [SELFIE_MODEL_URL];
}

async function createSegmenter(
  vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>,
  modelAssetPath: string
) {
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
  } catch (error) {
    console.warn('GPU delegate failed for selfie segmentation, falling back to CPU.', error);
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

function createEmptyBranch(
  width: number,
  height: number,
  labels: string[],
  ageMs: number
): SegmentationBranchResult {
  const pixelCount = width * height;
  return createBranchResult(
    width,
    height,
    new Uint8Array(pixelCount),
    new Float32Array(pixelCount),
    labels,
    ageMs
  );
}

function pickPersonConfidenceMask(confidenceMasks: MPMask[] | undefined, labels: readonly string[]) {
  if (!confidenceMasks?.length) return undefined;

  const personIndex = labels.findIndex((label) => label.toLowerCase().includes('person'));
  if (personIndex >= 0 && confidenceMasks[personIndex]) return confidenceMasks[personIndex];

  return confidenceMasks[confidenceMasks.length - 1] ?? confidenceMasks[0];
}

function resampleCategoryMask(
  source: Uint8Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number
) {
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

export class SegmentationManager {
  private humanSegmenter: SelfieSegmenterSlot | null = null;

  async initialize(_sourceWidth = 1280, _sourceHeight = 720): Promise<void> {
    if (this.humanSegmenter) return;

    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);

    for (const modelAssetPath of chooseSelfieModelCandidates()) {
      try {
        const segmenter = await createSegmenter(vision, modelAssetPath);
        this.humanSegmenter = {
          segmenter,
          labels: normalizeLabels(segmenter.getLabels())
        };
        break;
      } catch (error) {
        console.warn('Selfie segmentation model failed to initialize.', error);
      }
    }

    if (!this.humanSegmenter) {
      throw new Error('Unable to initialize the selfie segmentation model.');
    }
  }

  private segmentBranch(
    slot: SelfieSegmenterSlot,
    frame: ImageBitmap,
    timestampMs: number,
    ageMs = 0,
    outputWidth = frame.width,
    outputHeight = frame.height
  ): SegmentationBranchResult {
    const result = slot.segmenter.segmentForVideo(frame, timestampMs);
    const categoryMaskResult = result.categoryMask;

    if (!categoryMaskResult) {
      throw new Error('MediaPipe failed to return a human category mask.');
    }

    const categoryMask = extractBinaryMask(categoryMaskResult);
    if (!categoryMask) {
      throw new Error('MediaPipe failed to produce a usable selfie segmentation mask.');
    }

    const categoryWidth = categoryMaskResult.width;
    const categoryHeight = categoryMaskResult.height;
    const resizedCategoryMask = resampleCategoryMask(categoryMask, categoryWidth, categoryHeight, outputWidth, outputHeight);

    const personConfidenceMask = pickPersonConfidenceMask(result.confidenceMasks, slot.labels);
    const resizedConfidenceMask = personConfidenceMask
      ? resampleFloatMask(
        extractMaskFloats(personConfidenceMask),
        personConfidenceMask.width,
        personConfidenceMask.height,
        outputWidth,
        outputHeight
      )
      : Float32Array.from(resizedCategoryMask, (value) => value);

    categoryMaskResult.close();
    result.confidenceMasks?.forEach((mask) => mask.close());

    return createBranchResult(outputWidth, outputHeight, resizedCategoryMask, resizedConfidenceMask, slot.labels, ageMs);
  }

  async segment(frame: ImageBitmap, timestampMs: number): Promise<SegmentationFrameResult> {
    if (!this.humanSegmenter) {
      await this.initialize(frame.width, frame.height);
    }

    if (!this.humanSegmenter) {
      throw new Error('Segmentation model is not initialized.');
    }

    try {
      return {
        width: frame.width,
        height: frame.height,
        branches: [this.segmentBranch(this.humanSegmenter, frame, timestampMs, 0)]
      };
    } catch (error) {
      console.warn('Selfie segmentation frame failed, using an empty mask fallback.', error);
      return {
        width: frame.width,
        height: frame.height,
        branches: [createEmptyBranch(frame.width, frame.height, this.humanSegmenter.labels, 0)]
      };
    }
  }

  close() {
    this.humanSegmenter?.segmenter.close();
    this.humanSegmenter = null;
  }
}
