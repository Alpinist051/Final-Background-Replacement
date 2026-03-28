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

type LabelClass = 0 | 1 | 2 | 3;

type LabelProfile = {
  weights: Float32Array;
  classes: Uint8Array;
};

function normalizeLabel(label: string) {
  return label.trim().toLowerCase().replace(/[_-]+/g, ' ');
}

function isBackgroundLabel(label: string) {
  return /background/i.test(normalizeLabel(label));
}

function isHumanLabel(label: string) {
  const normalized = normalizeLabel(label);
  return /(person|human|selfie|body|torso|arm|leg|hand|foot|head|face|hair|skin|shirt|clothes|clothing|coat|jacket|dress|pants|skirt|sleeve|others?|accessor(y|ies)?|glasses|sunglasses)/i.test(normalized);
}

function isStrongSubjectLabel(label: string) {
  const normalized = normalizeLabel(label);
  return /(cat|dog|horse|cow|sheep|bird|car|bus|train|bicycle|motorbike|motorcycle|boat|chair|sofa|couch|table|dining table|plant|potted plant|tv|television|monitor|laptop|book|bottle|cup|bag|backpack|keyboard|mouse|phone|bench|airplane|aeroplane)/i.test(normalized);
}

function classifyLabel(label: string): LabelClass {
  if (isBackgroundLabel(label)) return 0;
  if (isHumanLabel(label)) return 1;
  if (isStrongSubjectLabel(label)) return 3;
  return 2;
}

function buildClassProfile(labels: string[]): LabelProfile {
  const size = Math.max(1, labels.length);
  const weights = new Float32Array(size);
  const classes = new Uint8Array(size);

  for (let i = 0; i < size; i += 1) {
    const labelClass = classifyLabel(labels[i] ?? '');
    classes[i] = labelClass;

    if (labelClass === 0) {
      weights[i] = 0;
    } else if (labelClass === 1) {
      weights[i] = 1.1;
    } else if (labelClass === 3) {
      weights[i] = 1.02;
    } else {
      weights[i] = 1.01;
    }
  }

  return { weights, classes };
}

export class MaskProcessor {
  private previousAlphaMask: Float32Array | null = null;
  private classProfileKey = '';
  private classProfile: LabelProfile = {
    weights: new Float32Array([0, 1]),
    classes: new Uint8Array([0, 2])
  };

  private getClassProfile(labels: string[]) {
    const key = labels.join('|');
    if (key !== this.classProfileKey) {
      this.classProfileKey = key;
      this.classProfile = buildClassProfile(labels);
    }
    return this.classProfile;
  }

  process(result: SegmentationFrameResult, tuning: VirtualBackgroundTuning, liveMotion = 0): ProcessedMask {
    const { width, height, branches } = result;
    const pixelCount = width * height;
    const motionFactor = clamp01(liveMotion * 5.5);
    const selfieAlpha = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount);
    const tuningBoost = Math.min(1.32, Math.max(0.94, tuning.confidenceBoost));

    for (const branch of branches) {
      if (branch.kind !== 'selfie') continue;

      const classProfile = this.getClassProfile(branch.labels);
      const sourceConfidence = branch.confidenceMask;

      for (let i = 0; i < pixelCount; i += 1) {
        const categoryIndex = branch.categoryMask[i] ?? 0;
        const labelClass = classProfile.classes[categoryIndex] ?? 0;
        if (labelClass === 0) continue;

        const confidence = sourceConfidence ? clamp01(sourceConfidence[i]) : 1;
        const classWeight = classProfile.weights[categoryIndex] ?? 1;
        let alpha = confidence * classWeight * tuningBoost;
        alpha = Math.max(alpha, confidence * 0.95);
        selfieAlpha[i] = Math.max(selfieAlpha[i], alpha);
        confidenceMask[i] = Math.max(confidenceMask[i], clamp01(confidence));
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
      const currentAlpha = selfieAlpha[i];
      let alpha = clamp01(currentAlpha);
      if (previousAlphaMask) {
        const stability = clamp01(confidenceMask[i] * 0.85 + (1 - motionFactor) * 0.2);
        if (motionFactor > 0.35) {
          const motionBlend = 0.82 + stability * 0.12;
          alpha = previousAlpha + (alpha - previousAlpha) * motionBlend;
        } else {
          const riseBlend = 0.42 + stability * 0.28;
          const fallBlend = 0.18 + stability * 0.3;
          alpha = alpha >= previousAlpha
            ? previousAlpha + (alpha - previousAlpha) * riseBlend
            : previousAlpha + (alpha - previousAlpha) * fallBlend;
        }
      }
      alpha = clamp01(alpha);
      nextAlphaMask[i] = alpha;

      if (previousAlphaMask && (Math.abs(currentAlpha - previousAlpha) > 0.08 || Math.abs(alpha - previousAlpha) > 0.12)) {
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
    this.classProfileKey = '';
    this.classProfile = {
      weights: new Float32Array([0, 1]),
      classes: new Uint8Array([0, 2])
    };
  }
}
