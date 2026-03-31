import type { SegmentationFrameResult, VirtualBackgroundTuning } from '@/types/engine';

export interface ProcessedMask {
  alphaMask: Float32Array;
  confidenceMask: Float32Array;
  motionMagnitude: number;
  foregroundRatio: number;
  maskMean: number;
  confidenceMean: number;
}

// Lower blend weights preserve more of the previous frame and reduce flicker.
const ALPHA_RISE_WEIGHT = 0.34;
const ALPHA_FALL_WEIGHT = 0.64;

function createFloatBuffer(length: number, fill = 0) {
  const buffer = new Float32Array(length);
  if (fill !== 0) buffer.fill(fill);
  return buffer;
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

export class MaskProcessor {
  private previousAlphaMask: Float32Array | null = null;

  process(result: SegmentationFrameResult, _tuning?: Pick<VirtualBackgroundTuning, 'confidenceBoost'>): ProcessedMask {
    const { width, height, branches } = result;
    const pixelCount = width * height;
    const alphaMask = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount);

    for (const branch of branches) {
      if (branch.kind !== 'human') continue;

      const sourceConfidence = branch.confidenceMask;
      for (let i = 0; i < pixelCount; i += 1) {
        const categoryValue = (branch.categoryMask[i] ?? 0) !== 0 ? 1 : 0;
        const confidence = sourceConfidence
          ? clamp01(sourceConfidence[i] ?? 0)
          : categoryValue;
        const personValue = categoryValue ? confidence : 0;
        const previousValue = this.previousAlphaMask?.[i] ?? 0;
        const targetValue = personValue;
        const blendWeight = targetValue >= previousValue ? ALPHA_RISE_WEIGHT : ALPHA_FALL_WEIGHT;
        const smoothedValue = previousValue + (targetValue - previousValue) * blendWeight;

        confidenceMask[i] = Math.max(confidenceMask[i], confidence);
        alphaMask[i] = Math.max(alphaMask[i], clamp01(smoothedValue));
      }
    }

    let motionMagnitude = 0;
    if (this.previousAlphaMask && this.previousAlphaMask.length === pixelCount) {
      let diffSum = 0;
      for (let i = 0; i < pixelCount; i += 1) {
        diffSum += Math.abs(alphaMask[i] - this.previousAlphaMask[i]);
      }
      motionMagnitude = diffSum / pixelCount;
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

    this.previousAlphaMask = new Float32Array(alphaMask);

    return {
      alphaMask,
      confidenceMask,
      motionMagnitude,
      foregroundRatio: foregroundPixels / pixelCount,
      maskMean: maskSum / pixelCount,
      confidenceMean: confidenceSum / pixelCount
    };
  }

  reset() {
    this.previousAlphaMask = null;
  }
}
