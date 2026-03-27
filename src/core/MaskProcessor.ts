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

function isBackgroundLabel(label: string) {
  return /background/i.test(label);
}

function isHumanLabel(label: string) {
  return /(person|human|selfie)/i.test(label);
}

function isSubjectLabel(label: string) {
  return !isBackgroundLabel(label) && !isHumanLabel(label);
}

function isStrongSubjectLabel(label: string) {
  return /(cat|dog|horse|cow|sheep|bird|car|bus|train|bicycle|motorbike|boat|chair|sofa|table|plant|potted plant|tv|monitor|laptop|book|bottle|cup|bag|backpack|keyboard|mouse|phone|bench)/i.test(label);
}

function buildClassWeights(kind: SegmentationFrameResult['branches'][number]['kind'], labels: string[]) {
  const weights = new Float32Array(Math.max(1, labels.length));
  for (let i = 0; i < weights.length; i += 1) {
    const label = labels[i]?.toLowerCase() ?? '';
    if (isBackgroundLabel(label)) {
      weights[i] = 0;
    } else if (kind === 'selfie') {
      weights[i] = 1.08;
    } else if (isStrongSubjectLabel(label)) {
      weights[i] = 1.05;
    } else {
      weights[i] = 1.0;
    }
  }
  return weights;
}

function branchBoost(kind: SegmentationFrameResult['branches'][number]['kind']) {
  return kind === 'selfie' ? 1.08 : 0.96;
}

function preserveFactor(kind: number) {
  return kind === 1 ? 0.84 : kind === 2 ? 0.76 : 0.72;
}

function isForegroundCategory(kind: SegmentationFrameResult['branches'][number]['kind'], categoryIndex: number, label: string) {
  if (categoryIndex === 0) return false;
  if (kind === 'selfie') return true;
  return !label || isSubjectLabel(label);
}

export class MaskProcessor {
  private previousAlphaMask: Float32Array | null = null;
  private previousDominantTypes: Uint8Array | null = null;
  private classWeightsKey = '';
  private classWeights: Float32Array = new Float32Array([0, 1]);

  private getClassWeights(kind: SegmentationFrameResult['branches'][number]['kind'], labels: string[]) {
    const key = `${kind}|${labels.join('|')}`;
    if (key !== this.classWeightsKey) {
      this.classWeightsKey = key;
      this.classWeights = buildClassWeights(kind, labels);
    }
    return this.classWeights;
  }

  process(result: SegmentationFrameResult, tuning: VirtualBackgroundTuning): ProcessedMask {
    const { width, height, branches } = result;
    const pixelCount = width * height;

    const rawAlpha = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount);
    const dominantTypes = new Uint8Array(pixelCount);
    const tuningBoost = Math.min(1.4, Math.max(0.9, tuning.confidenceBoost));

    for (const branch of branches) {
      const classWeights = this.getClassWeights(branch.kind, branch.labels);
      const kindBoost = branchBoost(branch.kind);
      const ageFactor = branch.kind === 'subject' ? Math.max(0.6, 1 - branch.ageMs / 480) : 1;
      const sourceConfidence = branch.confidenceMask;

      for (let i = 0; i < pixelCount; i += 1) {
        const categoryIndex = branch.categoryMask[i] ?? 0;
        const label = branch.labels[categoryIndex] ?? '';
        if (!isForegroundCategory(branch.kind, categoryIndex, label)) continue;

        const confidence = sourceConfidence ? clamp01(sourceConfidence[i]) : 1;
        const classWeight = classWeights[categoryIndex] ?? 1;
        let alpha = confidence * classWeight * kindBoost * ageFactor * tuningBoost;

        if (branch.kind === 'selfie') {
          alpha = Math.max(alpha, 0.88);
        } else if (isStrongSubjectLabel(label)) {
          alpha = Math.max(alpha, 0.84);
        }

        alpha = clamp01(alpha);
        if (alpha > rawAlpha[i]) {
          rawAlpha[i] = alpha;
          dominantTypes[i] = branch.kind === 'selfie' ? 1 : 2;
        }
        confidenceMask[i] = Math.max(confidenceMask[i], clamp01(confidence * ageFactor));
      }
    }

    const previousAlphaMask = this.previousAlphaMask;
    const previousDominantTypes = this.previousDominantTypes;
    const nextAlphaMask = createFloatBuffer(pixelCount);
    const nextDominantTypes = new Uint8Array(pixelCount);

    let motionMagnitude = 0;
    let foregroundPixels = 0;
    let alphaSum = 0;
    let confidenceSum = 0;

    for (let i = 0; i < pixelCount; i += 1) {
      const currentAlpha = rawAlpha[i];
      const previousAlpha = previousAlphaMask?.[i] ?? 0;
      const previousType = previousDominantTypes?.[i] ?? dominantTypes[i];

      let alpha = currentAlpha;
      if (previousAlpha > 0 && currentAlpha < previousAlpha) {
        alpha = Math.max(currentAlpha, previousAlpha * preserveFactor(previousType));
      }

      if (alpha > 0.06 && currentAlpha === 0 && previousAlpha > 0.08) {
        alpha = previousAlpha * 0.5;
      }

      nextAlphaMask[i] = clamp01(alpha);
      nextDominantTypes[i] = currentAlpha > 0 ? dominantTypes[i] : (previousAlpha > 0 ? previousType : 0);

      if (previousAlphaMask && Math.abs(currentAlpha - previousAlpha) > 0.15) {
        motionMagnitude += 1;
      }
      if (nextAlphaMask[i] > 0.12) foregroundPixels += 1;
      alphaSum += nextAlphaMask[i];
      confidenceSum += confidenceMask[i];
    }

    this.previousAlphaMask = new Float32Array(nextAlphaMask);
    this.previousDominantTypes = new Uint8Array(nextDominantTypes);

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
    this.previousDominantTypes = null;
    this.classWeightsKey = '';
    this.classWeights = new Float32Array([0, 1]);
  }
}
