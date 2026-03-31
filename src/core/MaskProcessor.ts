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

  process(result: SegmentationFrameResult, tuning: VirtualBackgroundTuning, liveMotion = 0): ProcessedMask {
    const { width, height, branches } = result;
    const pixelCount = width * height;
    const motionFactor = clamp01(liveMotion * (2.8 + tuning.motionBoost * 0.35));
    const foregroundAlpha = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount);
    const tuningBoost = Math.min(1.35, Math.max(0.92, tuning.confidenceBoost));

    for (const branch of branches) {
      if (branch.kind !== 'human') continue;

      const sourceConfidence = branch.confidenceMask;
      for (let i = 0; i < pixelCount; i += 1) {
        const isForeground = (branch.categoryMask[i] ?? 0) !== 0;
        if (!isForeground) continue;

        const confidence = sourceConfidence ? clamp01(sourceConfidence[i]) : 1;
        const alpha = clamp01(0.7 + confidence * 0.3 * tuningBoost);
        if (alpha >= foregroundAlpha[i]) {
          foregroundAlpha[i] = alpha;
        }
        confidenceMask[i] = Math.max(confidenceMask[i], confidence);
      }
    }

    const previousAlphaMask = this.previousAlphaMask;
    const nextAlphaMask = createFloatBuffer(pixelCount);

    let motionMagnitude = 0;
    let foregroundPixels = 0;
    let alphaSum = 0;
    let confidenceSum = 0;

    for (let i = 0; i < pixelCount; i += 1) {
      const previousAlpha = previousAlphaMask?.[i] ?? 0;
      const currentAlpha = foregroundAlpha[i];
      let alpha = clamp01(currentAlpha);

      if (previousAlphaMask) {
        const stability = clamp01(confidenceMask[i] * 0.82 + (1 - motionFactor) * 0.18);
        const riseBlend = clamp01(0.58 + stability * 0.32);
        const fallBlend = clamp01(0.16 + stability * 0.18);
        const carryBlend = clamp01(0.18 + stability * 0.5 + (1 - motionFactor) * 0.06);

        if (alpha >= previousAlpha) {
          alpha = previousAlpha + (alpha - previousAlpha) * riseBlend;
        } else {
          const softened = previousAlpha + (alpha - previousAlpha) * fallBlend;
          alpha = Math.max(softened, previousAlpha * carryBlend);
        }
      }

      alpha = clamp01(alpha);
      nextAlphaMask[i] = alpha;

      if (previousAlphaMask && (Math.abs(currentAlpha - previousAlpha) > 0.08 || Math.abs(alpha - previousAlpha) > 0.1)) {
        motionMagnitude += 1;
      }
      if (alpha > 0.12) foregroundPixels += 1;
      alphaSum += alpha;
      confidenceSum += confidenceMask[i];
    }

    this.previousAlphaMask = new Float32Array(nextAlphaMask);

    return {
      alphaMask: nextAlphaMask,
      confidenceMask,
      motionMagnitude: previousAlphaMask ? motionMagnitude / pixelCount : 0,
      foregroundRatio: foregroundPixels / pixelCount,
      maskMean: alphaSum / pixelCount,
      confidenceMean: confidenceSum / pixelCount
    };
  }

  reset() {
    this.previousAlphaMask = null;
  }
}
