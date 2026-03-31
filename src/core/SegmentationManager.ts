import {
  FilesetResolver,
  ImageSegmenter,
  PoseLandmarker,
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
const POSE_MODEL_FULL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task';
const POSE_MODEL_LITE_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';
const SELFIE_MODEL_SQUARE_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite';

const HUMAN_LABELS = ['background', 'person'];

type PoseSlot = {
  kind: 'pose';
  poseLandmarker: PoseLandmarker;
  labels: string[];
};

type SegmenterSlot = {
  kind: 'segmenter';
  segmenter: ImageSegmenter;
  labels: string[];
};

type HumanSlot = PoseSlot | SegmenterSlot;

function extractBinaryMask(mask: MPMask | undefined): Uint8Array | undefined {
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
    output[i] = (source[i] ?? 0) > 0 ? 1 : 0;
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

function choosePoseModelCandidates() {
  return [
    POSE_MODEL_LITE_URL,
    POSE_MODEL_FULL_URL
  ];
}

function chooseFallbackModelCandidates() {
  return [
    SELFIE_MODEL_SQUARE_URL
  ];
}

async function createPoseLandmarker(
  vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>,
  modelAssetPath: string
) {
  try {
    return await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: 'GPU' as const
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.55,
      minPosePresenceConfidence: 0.55,
      minTrackingConfidence: 0.6,
      outputSegmentationMasks: true
    });
  } catch (error) {
    console.warn('GPU delegate failed for pose human segmentation, falling back to CPU.', error);
    return PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: 'CPU' as const
      },
      runningMode: 'VIDEO',
      numPoses: 1,
      minPoseDetectionConfidence: 0.55,
      minPosePresenceConfidence: 0.55,
      minTrackingConfidence: 0.6,
      outputSegmentationMasks: true
    });
  }
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
  private humanSegmenter: HumanSlot | null = null;
  private fallbackSegmenter: SegmenterSlot | null = null;

  async initialize(_sourceWidth = 1280, _sourceHeight = 720): Promise<void> {
    if (this.humanSegmenter) return;

    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_URL);

    for (const modelAssetPath of choosePoseModelCandidates()) {
      try {
        const poseLandmarker = await createPoseLandmarker(vision, modelAssetPath);
        this.humanSegmenter = {
          kind: 'pose',
          poseLandmarker,
          labels: HUMAN_LABELS
        };
        break;
      } catch (error) {
        console.warn('Pose human segmentation model failed to initialize.', error);
      }
    }

    if (!this.humanSegmenter) {
      for (const modelAssetPath of chooseFallbackModelCandidates()) {
        try {
          const segmenter = await createSegmenter(vision, modelAssetPath);
          const labels = segmenter.getLabels();
          const resolvedLabels = labels.length > 0 ? labels : HUMAN_LABELS;
          this.humanSegmenter = {
            kind: 'segmenter',
            segmenter,
            labels: resolvedLabels
          };
          break;
        } catch (error) {
          console.warn('Human segmentation model failed to initialize.', error);
        }
      }
    } else {
      for (const modelAssetPath of chooseFallbackModelCandidates()) {
        try {
          const segmenter = await createSegmenter(vision, modelAssetPath);
          const labels = segmenter.getLabels();
          const resolvedLabels = labels.length > 0 ? labels : HUMAN_LABELS;
          this.fallbackSegmenter = {
            kind: 'segmenter',
            segmenter,
            labels: resolvedLabels
          };
          break;
        } catch (error) {
          console.warn('Fallback human segmentation model failed to initialize.', error);
        }
      }
    }

    if (!this.humanSegmenter) {
      throw new Error('Unable to initialize the human segmentation model.');
    }
  }

  private segmentBranch(
    slot: HumanSlot,
    frame: ImageBitmap,
    timestampMs: number,
    ageMs = 0,
    outputWidth = frame.width,
    outputHeight = frame.height
  ): SegmentationBranchResult {
    const sourceWidth = frame.width;
    const sourceHeight = frame.height;

    if (slot.kind === 'pose') {
      const result = slot.poseLandmarker.detectForVideo(frame, timestampMs);
      const segmentationMask = result.segmentationMasks?.[0];

      if (!segmentationMask) {
        throw new Error('Pose landmarker failed to return a segmentation mask.');
      }

      const categoryMask = extractBinaryMask(segmentationMask);
      const confidenceMask = extractMaskFloats(segmentationMask);
      const resizedCategoryMask = resampleCategoryMask(categoryMask ?? new Uint8Array(sourceWidth * sourceHeight), sourceWidth, sourceHeight, outputWidth, outputHeight);
      const resizedConfidenceMask = resampleFloatMask(confidenceMask, sourceWidth, sourceHeight, outputWidth, outputHeight);

      result.segmentationMasks?.forEach((mask) => mask.close());

      return createBranchResult(outputWidth, outputHeight, resizedCategoryMask, resizedConfidenceMask, slot.labels, ageMs);
    }

    const result = slot.segmenter.segmentForVideo(frame, timestampMs);
    const categoryMask = extractBinaryMask(result.categoryMask);
    const confidenceMask = result.confidenceMasks?.[1] ?? result.confidenceMasks?.[0];

    if (!categoryMask) {
      throw new Error('MediaPipe failed to return a human category mask.');
    }

    const resizedCategoryMask = resampleCategoryMask(categoryMask, sourceWidth, sourceHeight, outputWidth, outputHeight);
    const resizedConfidenceMask = confidenceMask
      ? resampleFloatMask(extractMaskFloats(confidenceMask), sourceWidth, sourceHeight, outputWidth, outputHeight)
      : undefined;

    result.categoryMask?.close();
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

    const branches: SegmentationBranchResult[] = [];

    try {
      branches.push(this.segmentBranch(this.humanSegmenter, frame, timestampMs, 0));
    } catch (error) {
      console.warn('Human segmentation frame failed.', error);
    }

    if (!branches.length && this.fallbackSegmenter) {
      try {
        branches.push(this.segmentBranch(this.fallbackSegmenter, frame, timestampMs, 1));
      } catch (error) {
        console.warn('Fallback human segmentation frame failed.', error);
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
    if (this.humanSegmenter?.kind === 'pose') {
      this.humanSegmenter.poseLandmarker.close();
    } else {
      this.humanSegmenter?.segmenter.close();
    }
    this.fallbackSegmenter?.segmenter.close();
    this.humanSegmenter = null;
    this.fallbackSegmenter = null;
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
