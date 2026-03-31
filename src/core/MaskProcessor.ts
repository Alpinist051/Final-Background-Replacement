import type { SegmentationFrameResult, VirtualBackgroundTuning } from '@/types/engine';

export interface ProcessedMask {
  alphaMask: Float32Array;
  confidenceMask: Float32Array;
  motionMagnitude: number;
  foregroundRatio: number;
  maskMean: number;
  confidenceMean: number;
}

// Favor the current frame on rising edges so the person stays visible,
// but keep a little memory on falling edges to reduce flicker.
const ALPHA_RISE_WEIGHT = 0.82;
const ALPHA_FALL_WEIGHT = 0.46;
const PERSON_ALPHA_FLOOR = 0.48;
const PERSON_FILL_RADIUS = 2;
const PERSON_FILL_THRESHOLD = 0.18;
const PERSON_FILL_STRENGTH = 1;
const STRONG_FOREGROUND_THRESHOLD = 0.68;
const SUPPORT_CONFIDENCE_THRESHOLD = 0.34;
const SUPPORT_NEIGHBOR_THRESHOLD = 2;
const HOLE_NEIGHBOR_THRESHOLD = 5;

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
    const confidenceBoost = Math.max(1, tuning?.confidenceBoost ?? 1);

    for (const branch of branches) {
      if (branch.kind !== 'human') continue;

      const sourceConfidence = branch.confidenceMask;
      for (let i = 0; i < pixelCount; i += 1) {
        const categoryValue = (branch.categoryMask[i] ?? 0) !== 0 ? 1 : 0;
        const confidence = sourceConfidence
          ? clamp01(sourceConfidence[i] ?? 0)
          : categoryValue;
        const boostedConfidence = clamp01(confidence * confidenceBoost);
        const personValue = sourceConfidence
          ? clamp01(Math.max(boostedConfidence, categoryValue ? PERSON_ALPHA_FLOOR : 0))
          : categoryValue;
        const previousValue = this.previousAlphaMask?.[i] ?? 0;
        const targetValue = personValue;
        const blendWeight = targetValue >= previousValue ? ALPHA_RISE_WEIGHT : ALPHA_FALL_WEIGHT;
        const smoothedValue = previousValue + (targetValue - previousValue) * blendWeight;

        confidenceMask[i] = Math.max(confidenceMask[i], confidence);
        alphaMask[i] = Math.max(alphaMask[i], clamp01(smoothedValue));
      }
    }

    const refinedAlphaMask = createFloatBuffer(pixelCount);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let localMax = 0;
        for (let offsetY = -PERSON_FILL_RADIUS; offsetY <= PERSON_FILL_RADIUS; offsetY += 1) {
          const sampleY = Math.max(0, Math.min(height - 1, y + offsetY));
          const rowOffset = sampleY * width;
          for (let offsetX = -PERSON_FILL_RADIUS; offsetX <= PERSON_FILL_RADIUS; offsetX += 1) {
            const sampleX = Math.max(0, Math.min(width - 1, x + offsetX));
            const sampleValue = alphaMask[rowOffset + sampleX] ?? 0;
            if (sampleValue > localMax) localMax = sampleValue;
          }
        }

        const index = y * width + x;
        const currentValue = alphaMask[index] ?? 0;
        const fillValue = localMax >= PERSON_FILL_THRESHOLD ? localMax * PERSON_FILL_STRENGTH : currentValue;
        refinedAlphaMask[index] = Math.max(currentValue, fillValue);
      }
    }

    const supportedAlphaMask = createFloatBuffer(pixelCount);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const index = y * width + x;
        const currentValue = refinedAlphaMask[index] ?? 0;
        const currentConfidence = confidenceMask[index] ?? 0;

        let strongNeighbors = 0;
        for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
          const sampleY = Math.max(0, Math.min(height - 1, y + offsetY));
          const rowOffset = sampleY * width;
          for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
            if (offsetX === 0 && offsetY === 0) continue;
            const sampleX = Math.max(0, Math.min(width - 1, x + offsetX));
            if ((refinedAlphaMask[rowOffset + sampleX] ?? 0) >= STRONG_FOREGROUND_THRESHOLD) {
              strongNeighbors += 1;
            }
          }
        }

        if (currentValue >= STRONG_FOREGROUND_THRESHOLD) {
          supportedAlphaMask[index] = 1;
          continue;
        }

        if (currentConfidence >= SUPPORT_CONFIDENCE_THRESHOLD && strongNeighbors >= SUPPORT_NEIGHBOR_THRESHOLD) {
          supportedAlphaMask[index] = 1;
          continue;
        }

        if (currentConfidence >= PERSON_FILL_THRESHOLD && strongNeighbors >= HOLE_NEIGHBOR_THRESHOLD) {
          supportedAlphaMask[index] = 1;
          continue;
        }

        supportedAlphaMask[index] = currentValue >= 0.5 ? currentValue : 0;
      }
    }

    let motionMagnitude = 0;
    if (this.previousAlphaMask && this.previousAlphaMask.length === pixelCount) {
      let diffSum = 0;
      for (let i = 0; i < pixelCount; i += 1) {
        diffSum += Math.abs(supportedAlphaMask[i] - this.previousAlphaMask[i]);
      }
      motionMagnitude = diffSum / pixelCount;
    }

    let foregroundPixels = 0;
    let maskSum = 0;
    let confidenceSum = 0;
    for (let i = 0; i < pixelCount; i += 1) {
      const alpha = supportedAlphaMask[i];
      const confidence = confidenceMask[i];
      if (alpha > 0.5) foregroundPixels += 1;
      maskSum += alpha;
      confidenceSum += confidence;
    }

    this.previousAlphaMask = new Float32Array(supportedAlphaMask);

    return {
      alphaMask: supportedAlphaMask,
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
