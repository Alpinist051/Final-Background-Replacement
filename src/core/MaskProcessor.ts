import type { SegmentationFrameResult, VirtualBackgroundTuning } from '@/types/engine';

export interface ProcessedMask {
  alphaMask: Float32Array;
  confidenceMask: Float32Array;
  motionMagnitude: number;
  foregroundRatio: number;
  maskMean: number;
  confidenceMean: number;
}

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

  process(result: SegmentationFrameResult, tuning?: Pick<VirtualBackgroundTuning, 'confidenceBoost'>): ProcessedMask {
    const { width, height, branches } = result;
    const pixelCount = width * height;
    const alphaMask = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount);
    const confidenceBoost = Math.max(0.25, tuning?.confidenceBoost ?? 1);

    for (const branch of branches) {
      if (branch.kind !== 'human') continue;

      const sourceConfidence = branch.confidenceMask;
      for (let i = 0; i < pixelCount; i += 1) {
        const confidence = sourceConfidence ? clamp01(sourceConfidence[i] ?? 0) : ((branch.categoryMask[i] ?? 0) !== 0 ? 1 : 0);
        const boostedConfidence = sourceConfidence ? clamp01(Math.pow(confidence, 1 / confidenceBoost)) : confidence;
        alphaMask[i] = Math.max(alphaMask[i], boostedConfidence);
        confidenceMask[i] = Math.max(confidenceMask[i], confidence);
      }
    }

    let motionMagnitude = 0;
    const previousAlphaMask = this.previousAlphaMask;
    if (previousAlphaMask && previousAlphaMask.length === pixelCount) {
      let changedPixels = 0;
      for (let i = 0; i < pixelCount; i += 1) {
        if ((alphaMask[i] > 0.5) !== (previousAlphaMask[i] > 0.5)) {
          changedPixels += 1;
        }
      }
      motionMagnitude = changedPixels / pixelCount;
    }

    let foregroundPixels = 0;
    let maskSum = 0;
    for (let i = 0; i < pixelCount; i += 1) {
      const value = alphaMask[i];
      if (value > 0.5) foregroundPixels += 1;
      maskSum += value;
    }

    this.previousAlphaMask = new Float32Array(alphaMask);

    return {
      alphaMask,
      confidenceMask,
      motionMagnitude,
      foregroundRatio: foregroundPixels / pixelCount,
      maskMean: maskSum / pixelCount,
      confidenceMean: confidenceMask.reduce((sum, value) => sum + value, 0) / pixelCount
    };
  }

  reset() {
    this.previousAlphaMask = null;
  }
}
