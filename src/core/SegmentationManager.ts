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
const SELFIE_MODEL_SQUARE_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';

type SegmenterSlot = {
  segmenter: ImageSegmenter;
  labels: string[];
  humanIndex: number;
};

function extractCategoryMask(mask: MPMask | undefined): Uint8Array | undefined {
  if (!mask) return undefined;
  if (mask.hasUint8Array()) {
    return mask.getAsUint8Array();
  }

  const source = mask.getAsFloat32Array();
  const output = new Uint8Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    output[i] = source[i] >= 0.5 ? 1 : 0;
  }
  return output;
}

function extractMaskFloats(mask: MPMask): Float32Array {
  return mask.hasFloat32Array()
    ? mask.getAsFloat32Array()
    : Float32Array.from(mask.getAsUint8Array(), (value) => value / 255);
}

function normalizeLabel(label: string) {
  return label.trim().toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ');
}

function isBackgroundLabel(label: string) {
  return normalizeLabel(label).includes('background');
}

function isHumanLabel(label: string) {
  const normalized = normalizeLabel(label);
  return normalized.includes('person') || normalized.includes('human');
}

function getHumanIndex(labels: string[]) {
  const humanIndex = labels.findIndex((label) => isHumanLabel(label));
  if (humanIndex >= 0) return humanIndex;

  const backgroundIndex = labels.findIndex((label) => isBackgroundLabel(label));
  if (backgroundIndex === 0 && labels.length > 1) return 1;
  return labels.length > 1 ? labels.length - 1 : 0;
}

function chooseHumanModelCandidates() {
  return [SELFIE_MODEL_SQUARE_URL];
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
    console.warn('GPU delegate failed for human segmentation, falling back to CPU.', error);
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

  async initialize(sourceWidth = 1280, sourceHeight = 720): Promise<void> {
    if (this.humanSegmenter) return;

    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);
    const humanModelCandidates = chooseHumanModelCandidates();

    for (const modelAssetPath of humanModelCandidates) {
      try {
        const humanSegmenter = await createSegmenter(vision, modelAssetPath);
        const labels = humanSegmenter.getLabels();
        this.humanSegmenter = {
          segmenter: humanSegmenter,
          labels,
          humanIndex: getHumanIndex(labels)
        };
        break;
      } catch (error) {
        console.warn('Human segmentation model failed to initialize.', error);
      }
    }

    if (!this.humanSegmenter) {
      throw new Error('Unable to initialize the human segmentation model.');
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
    const confidenceMasks = result.confidenceMasks;

    let categoryMask = extractCategoryMask(result.categoryMask);
    const humanConfidenceMask = confidenceMasks?.[slot.humanIndex]
      ?? (confidenceMasks && confidenceMasks.length > 0
        ? confidenceMasks[confidenceMasks.length - 1]
        : undefined);
    const confidenceMask = humanConfidenceMask ? extractMaskFloats(humanConfidenceMask) : undefined;

    if (!categoryMask && confidenceMask) {
      categoryMask = new Uint8Array(confidenceMask.length);
      for (let i = 0; i < confidenceMask.length; i += 1) {
        categoryMask[i] = confidenceMask[i] >= 0.5 ? 1 : 0;
      }
    }

    if (!categoryMask) {
      throw new Error('MediaPipe failed to return a human category mask.');
    }

    const resizedCategoryMask = resampleCategoryMask(categoryMask, sourceWidth, sourceHeight, outputWidth, outputHeight);
    const resizedConfidenceMask = confidenceMask
      ? resampleConfidenceMask(confidenceMask, sourceWidth, sourceHeight, outputWidth, outputHeight)
      : undefined;

    const branch = createBranchResult(outputWidth, outputHeight, resizedCategoryMask, resizedConfidenceMask, slot.labels, ageMs);

    result.categoryMask?.close();
    confidenceMasks?.forEach((mask) => mask.close());

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

    try {
      branches.push(this.segmentBranch(this.humanSegmenter, frame, timestampMs, 0));
    } catch (error) {
      console.warn('Human segmentation frame failed.', error);
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

function resampleConfidenceMask(
  source: Float32Array,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number
) {
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
