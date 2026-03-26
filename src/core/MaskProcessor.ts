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

function isBackgroundLabel(label: string) { return /background/i.test(label); }
function isHairLabel(label: string) { return /hair/i.test(label); }
function isClothesOrSoftLabel(label: string) {
  return /(cloth|clothes|shirt|jacket|sleeve|upper|torso|body|person|arm|hand|leg|foot|pants|trouser|dress|skirt|shoe|others|other|accessory|object|headset|cup|paper)/i.test(label);
}

function isStrongLabel(label: string) {
  return /(hair|face|body|person|skin|neck|torso|cloth|clothes|shirt|jacket|sleeve|upper)/i.test(label);
}

function buildClassWeights(labels: string[]) {
  const weights = new Float32Array(Math.max(1, labels.length));
  for (let i = 0; i < weights.length; i++) {
    const label = labels[i]?.toLowerCase() ?? '';
    if (isBackgroundLabel(label)) {
      weights[i] = 0;
    } else if (isHairLabel(label)) {
      weights[i] = 1.12;
    } else if (isClothesOrSoftLabel(label)) {
      weights[i] = 1.15;
    } else {
      weights[i] = 1.0;
    }
  }
  return weights;
}

export class MaskProcessor {
  private previousCategoryMask: Uint8Array | null = null;
  private previousAlphaMask: Float32Array | null = null;
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

    for (let i = 0; i < pixelCount; i++) {
      const cat = categoryMask[i];
      const confidence = rawConfidence ? Math.max(0, Math.min(1, rawConfidence[i])) : 1;
      const classWeight = classWeights[cat] ?? (cat === 0 ? 0 : 0.98);

      if (cat === 0) {
        alphaMask[i] = 0;
      } else {
        let alpha = confidence * classWeight * Math.min(1.5, Math.max(0.95, tuning.confidenceBoost));

        if (isClothesOrSoftLabel(labels[cat] ?? '')) {
          alpha = Math.max(alpha, 0.96);
        } else if (isHairLabel(labels[cat] ?? '')) {
          alpha = Math.max(alpha, 0.97);
        }

        alphaMask[i] = Math.min(1, alpha);
      }

      confidenceMask[i] = confidence;
      alphaSum += alphaMask[i];
      confidenceSum += confidence;
    }

    const previousAlphaMask = this.previousAlphaMask;
    const nextAlphaMask = createFloatBuffer(pixelCount);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const index = y * width + x;
        const currentAlpha = alphaMask[index];
        const currentLabel = labels[categoryMask[index]] ?? '';
        const previousAlpha = previousAlphaMask?.[index] ?? 0;

        let alpha = currentAlpha;
        if (previousAlpha > 0) {
          const preserve = isHairLabel(currentLabel) ? 0.86 : isClothesOrSoftLabel(currentLabel) ? 0.8 : 0.72;
          if (currentAlpha < previousAlpha) {
            alpha = Math.max(currentAlpha, previousAlpha * preserve);
          }
        }

        if (alpha > 0.06 && currentAlpha === 0 && previousAlpha > 0.08) {
          alpha = previousAlpha * 0.5;
        }

        nextAlphaMask[index] = Math.min(1, alpha);
        if (alpha > 0.12) foregroundPixels += 1;
      }
    }

    alphaMask.set(nextAlphaMask);

    let motionMagnitude = 0;
    if (this.previousCategoryMask) {
      let diff = 0;
      for (let i = 0; i < pixelCount; i++) {
        if (categoryMask[i] !== this.previousCategoryMask[i]) diff++;
      }
      motionMagnitude = diff / pixelCount;
    }

    this.previousCategoryMask = new Uint8Array(categoryMask);
    this.previousAlphaMask = new Float32Array(alphaMask);

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
    this.previousAlphaMask = null;
    this.classWeightsKey = '';
    this.classWeights = new Float32Array([0, 1]);
  }
}
