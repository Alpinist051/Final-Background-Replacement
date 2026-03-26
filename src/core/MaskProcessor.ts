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

function isBackgroundLabel(label: string) {
  return /background/i.test(label);
}

function isHairLabel(label: string) {
  return /hair/i.test(label);
}

function isStrongForegroundLabel(label: string) {
  return /(face|body|person|skin|neck|torso|cloth|clothes|shirt|jacket|sleeve|upper)/i.test(label);
}

function isSoftForegroundLabel(label: string) {
  return /(others|other|accessory|glasses|headset|object)/i.test(label);
}

function buildClassWeights(labels: string[]) {
  const weights = new Float32Array(Math.max(1, labels.length));
  for (let i = 0; i < weights.length; i += 1) {
    const label = labels[i]?.toLowerCase?.() ?? '';
    if (isBackgroundLabel(label)) {
      weights[i] = 0;
    } else if (isHairLabel(label)) {
      weights[i] = 1.05;
    } else if (isStrongForegroundLabel(label)) {
      weights[i] = 1.0;
    } else if (isSoftForegroundLabel(label)) {
      weights[i] = 0.92;
    } else {
      weights[i] = 0.95;
    }
  }
  return weights;
}

export class MaskProcessor {
  private previousCategoryMask: Uint8Array | null = null;
  private classWeightsKey = '';
  private classWeights: Float32Array = new Float32Array([0, 1]);

  private getClassWeights(labels: string[]) {
    const key = labels.join('|');
    if (key !== this.classWeightsKey) {
      this.classWeightsKey = key;
      this.classWeights = buildClassWeights(labels);
    }
    return this.classWeights;
  }

  process(result: SegmentationFrameResult, tuning: VirtualBackgroundTuning): ProcessedMask {
    const { width, height, categoryMask, confidenceMask: rawConfidence, labels } = result;
    const pixelCount = width * height;
    const classWeights = this.getClassWeights(labels);

    const alphaMask = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount, 1.0);
    let foregroundPixels = 0;
    let alphaSum = 0;
    let confidenceSum = 0;

    for (let i = 0; i < pixelCount; i += 1) {
      const cat = categoryMask[i];
      const confidence = rawConfidence ? Math.max(0, Math.min(1, rawConfidence[i])) : 1;
      const classWeight = classWeights[cat] ?? (cat === 0 ? 0 : 0.95);

      if (cat === 0) {
        alphaMask[i] = 0;
      } else {
        const weightedAlpha = confidence * classWeight * Math.min(1.25, Math.max(0.85, tuning.confidenceBoost));
        const label = labels[cat] ?? '';
        const hardenedAlpha = isHairLabel(label) ? Math.max(weightedAlpha, 0.94) : Math.max(weightedAlpha, 0.88);
        alphaMask[i] = Math.min(1, hardenedAlpha);
        foregroundPixels += 1;
      }

      confidenceMask[i] = confidence;
      alphaSum += alphaMask[i];
      confidenceSum += confidence;
    }

    let motionMagnitude = 0;
    if (this.previousCategoryMask) {
      let diff = 0;
      for (let i = 0; i < pixelCount; i += 1) {
        if (categoryMask[i] !== this.previousCategoryMask[i]) diff += 1;
      }
      motionMagnitude = diff / pixelCount;
    }

    this.previousCategoryMask = new Uint8Array(categoryMask);

    return {
      alphaMask,
      confidenceMask,
      motionMagnitude,
      foregroundRatio: foregroundPixels / pixelCount,
      maskMean: alphaSum / pixelCount,
      confidenceMean: confidenceSum / pixelCount
    };
  }

  reset() {
    this.previousCategoryMask = null;
    this.classWeightsKey = '';
    this.classWeights = new Float32Array([0, 1]);
  }
}
