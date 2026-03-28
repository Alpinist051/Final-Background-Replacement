import type { SegmentationFrameResult, VirtualBackgroundTuning } from '@/types/engine';
import { buildSemanticProfile, type SemanticProfile } from './segmentationLabels';

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
  private semanticProfileKey = '';
  private semanticProfile: SemanticProfile = buildSemanticProfile([
    'background',
    'hair',
    'body-skin',
    'face-skin',
    'clothes',
    'others'
  ]);

  private getSemanticProfile(labels: string[]) {
    const key = labels.join('|');
    if (key !== this.semanticProfileKey) {
      this.semanticProfileKey = key;
      this.semanticProfile = buildSemanticProfile(labels);
    }
    return this.semanticProfile;
  }

  process(result: SegmentationFrameResult, tuning: VirtualBackgroundTuning, liveMotion = 0): ProcessedMask {
    const { width, height, branches } = result;
    const pixelCount = width * height;
    const motionFactor = clamp01(liveMotion * 3.2);
    const foregroundAlpha = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount);
    const semanticClasses = new Uint8Array(pixelCount);
    const tuningBoost = Math.min(1.28, Math.max(0.94, tuning.confidenceBoost));

    for (const branch of branches) {
      if (branch.kind !== 'human') continue;

      const semanticProfile = this.getSemanticProfile(branch.labels);
      const sourceConfidence = branch.confidenceMask;

      for (let i = 0; i < pixelCount; i += 1) {
        const categoryIndex = branch.categoryMask[i] ?? 0;
        const labelClass = semanticProfile.labelClasses[categoryIndex] ?? 0;
        if (labelClass === 0) continue;

        const confidence = sourceConfidence ? clamp01(sourceConfidence[i]) : 1;
        const alphaWeight = semanticProfile.labelAlphaWeights[categoryIndex] ?? 1;
        const confidenceWeight = semanticProfile.labelConfidenceWeights[categoryIndex] ?? 1;
        const alpha = clamp01(Math.max(confidence * alphaWeight * tuningBoost, confidence * 0.92));
        if (alpha >= foregroundAlpha[i]) {
          foregroundAlpha[i] = alpha;
          semanticClasses[i] = labelClass;
        }
        confidenceMask[i] = Math.max(confidenceMask[i], clamp01(confidence * confidenceWeight));
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
      const labelClass = semanticClasses[i] ?? 0;
      const riseWeight = this.semanticProfile.classRiseWeights[labelClass] ?? 0.5;
      const fallWeight = this.semanticProfile.classFallWeights[labelClass] ?? 0.2;
      const carryWeight = this.semanticProfile.classCarryWeights[labelClass] ?? 0.8;
      let alpha = clamp01(currentAlpha);
      if (previousAlphaMask) {
        const stability = clamp01(confidenceMask[i] * 0.78 + carryWeight * 0.22 + (1 - motionFactor) * 0.08);
        const riseBlend = clamp01(riseWeight * (0.56 + stability * 0.34));
        const fallBlend = clamp01(fallWeight * (0.46 + stability * 0.30));
        const carryBlend = clamp01(0.18 + carryWeight * 0.46 + (1 - motionFactor) * 0.06);
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
    this.semanticProfileKey = '';
    this.semanticProfile = buildSemanticProfile([
      'background',
      'hair',
      'body-skin',
      'face-skin',
      'clothes',
      'others'
    ]);
  }
}
