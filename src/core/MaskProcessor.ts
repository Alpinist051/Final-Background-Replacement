import type { SegmentationFrameResult, VirtualBackgroundTuning } from '@/types/engine';

export interface ProcessedMask {
  alphaMask: Float32Array;
  confidenceMask: Float32Array;
  motionMagnitude: number;
  foregroundRatio: number;
  maskMean: number;
  confidenceMean: number;
}

const PERSON_THRESHOLD = 0.65;

function createFloatBuffer(length: number, fill = 0) {
  const buffer = new Float32Array(length);
  if (fill !== 0) buffer.fill(fill);
  return buffer;
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

export class MaskProcessor {
  process(result: SegmentationFrameResult, _tuning?: Pick<VirtualBackgroundTuning, 'confidenceBoost'>): ProcessedMask {
    const { width, height, branches } = result;
    const pixelCount = width * height;
    const alphaMask = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount);

    for (const branch of branches) {
      if (branch.kind !== 'human') continue;

      const sourceConfidence = branch.confidenceMask;
      for (let i = 0; i < pixelCount; i += 1) {
        const confidence = sourceConfidence
          ? clamp01(sourceConfidence[i] ?? 0)
          : ((branch.categoryMask[i] ?? 0) !== 0 ? 1 : 0);
        const personValue = confidence >= PERSON_THRESHOLD ? confidence : 0;

        confidenceMask[i] = Math.max(confidenceMask[i], confidence);
        alphaMask[i] = Math.max(alphaMask[i], personValue);
      }
    }

    let foregroundPixels = 0;
    let maskSum = 0;
    let confidenceSum = 0;
    for (let i = 0; i < pixelCount; i += 1) {
      const alpha = alphaMask[i];
      const confidence = confidenceMask[i];
      if (alpha > 0.5) foregroundPixels += 1;
      maskSum += alpha;
      confidenceSum += confidence;
    }

    return {
      alphaMask,
      confidenceMask,
      motionMagnitude: 0,
      foregroundRatio: foregroundPixels / pixelCount,
      maskMean: maskSum / pixelCount,
      confidenceMean: confidenceSum / pixelCount
    };
  }

  reset() {}
}
