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

const HUMAN_LABELS = ['background', 'person'];

type SegmenterSlot = {
  segmenter: ImageSegmenter;
};

function extractCategoryMask(mask: MPMask | undefined): Uint8Array | undefined {
  if (!mask) return undefined;

  if (mask.hasUint8Array()) {
    const source = mask.getAsUint8Array();
    const output = new Uint8Array(source.length);
    for (let i = 0; i < source.length; i += 1) {
      output[i] = source[i] > 0 ? 1 : 0;
    }
    return output;
  }

  const source = mask.getAsFloat32Array();
  const output = new Uint8Array(source.length);
  for (let i = 0; i < source.length; i += 1) {
    output[i] = source[i] >= 0.5 ? 1 : 0;
  }
  return output;
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
      outputConfidenceMasks: false
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
      outputConfidenceMasks: false
    });
  }
}

function createBranchResult(
  width: number,
  height: number,
  categoryMask: Uint8Array,
  ageMs: number
): SegmentationBranchResult {
  return {
    kind: 'human',
    width,
    height,
    categoryMask,
    labels: HUMAN_LABELS,
    ageMs
  };
}

export class SegmentationManager {
  private humanSegmenter: SegmenterSlot | null = null;

  async initialize(_sourceWidth = 1280, _sourceHeight = 720): Promise<void> {
    if (this.humanSegmenter) return;

    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);

    for (const modelAssetPath of chooseHumanModelCandidates()) {
      try {
        const segmenter = await createSegmenter(vision, modelAssetPath);
        this.humanSegmenter = { segmenter };
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
    const categoryMask = extractCategoryMask(result.categoryMask);

    if (!categoryMask) {
      throw new Error('MediaPipe failed to return a human category mask.');
    }

    const resizedCategoryMask = resampleCategoryMask(categoryMask, sourceWidth, sourceHeight, outputWidth, outputHeight);

    result.categoryMask?.close();
    result.confidenceMasks?.forEach((mask) => mask.close());

    return createBranchResult(outputWidth, outputHeight, resizedCategoryMask, ageMs);
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
