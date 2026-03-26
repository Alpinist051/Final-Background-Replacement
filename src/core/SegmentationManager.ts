import {
  FilesetResolver,
  ImageSegmenter,
  type MPMask
} from '@mediapipe/tasks-vision';
import type { SegmentationFrameResult } from '@/types/engine';

const VISION_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite';

function extractMask(mask: MPMask | undefined): Uint8Array | undefined {
  if (!mask) return undefined;
  return mask.hasUint8Array() ? mask.getAsUint8Array() : new Uint8Array(mask.getAsFloat32Array());
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

    const categoryMask = extractMask(result.categoryMask);
    const confidenceMaskRaw = result.confidenceMasks?.[0] ? extractMask(result.confidenceMasks[0]) : undefined;

    if (!categoryMask || !(categoryMask instanceof Uint8Array)) {
      throw new Error('MediaPipe failed to return category mask');
    }

    const output: SegmentationFrameResult = {
      width: frame.width,   // force exact input resolution (super-perfect alignment)
      height: frame.height,
      categoryMask,
      confidenceMask: confidenceMaskRaw instanceof Float32Array
        ? confidenceMaskRaw
        : confidenceMaskRaw ? new Float32Array(confidenceMaskRaw) : undefined,
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