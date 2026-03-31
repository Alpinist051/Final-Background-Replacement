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

function applyMaxFilter3x3(source: Float32Array, width: number, height: number) {
  const output = new Float32Array(source.length);
  for (let y = 0; y < height; y += 1) {
    const y0 = Math.max(0, y - 1);
    const y1 = Math.min(height - 1, y + 1);
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      const x0 = Math.max(0, x - 1);
      const x1 = Math.min(width - 1, x + 1);
      let maxValue = 0;
      for (let ny = y0; ny <= y1; ny += 1) {
        const sourceRow = ny * width;
        for (let nx = x0; nx <= x1; nx += 1) {
          maxValue = Math.max(maxValue, source[sourceRow + nx] ?? 0);
        }
      }
      output[row + x] = maxValue;
    }
  }
  return output;
}

function applyMinFilter3x3(source: Float32Array, width: number, height: number) {
  const output = new Float32Array(source.length);
  for (let y = 0; y < height; y += 1) {
    const y0 = Math.max(0, y - 1);
    const y1 = Math.min(height - 1, y + 1);
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      const x0 = Math.max(0, x - 1);
      const x1 = Math.min(width - 1, x + 1);
      let minValue = 1;
      for (let ny = y0; ny <= y1; ny += 1) {
        const sourceRow = ny * width;
        for (let nx = x0; nx <= x1; nx += 1) {
          minValue = Math.min(minValue, source[sourceRow + nx] ?? 0);
        }
      }
      output[row + x] = minValue;
    }
  }
  return output;
}

function applyBoxBlur3x3(source: Float32Array, width: number, height: number) {
  const output = new Float32Array(source.length);
  for (let y = 0; y < height; y += 1) {
    const y0 = Math.max(0, y - 1);
    const y1 = Math.min(height - 1, y + 1);
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      const x0 = Math.max(0, x - 1);
      const x1 = Math.min(width - 1, x + 1);
      let sum = 0;
      let count = 0;
      for (let ny = y0; ny <= y1; ny += 1) {
        const sourceRow = ny * width;
        for (let nx = x0; nx <= x1; nx += 1) {
          sum += source[sourceRow + nx] ?? 0;
          count += 1;
        }
      }
      output[row + x] = count > 0 ? sum / count : 0;
    }
  }
  return output;
}

function refineAlphaMask(
  rawAlphaMask: Float32Array,
  confidenceMask: Float32Array,
  width: number,
  height: number
) {
  const expanded = applyMaxFilter3x3(rawAlphaMask, width, height);
  const closed = applyMinFilter3x3(expanded, width, height);
  const blurred = applyBoxBlur3x3(closed, width, height);
  const output = new Float32Array(rawAlphaMask.length);

  for (let i = 0; i < rawAlphaMask.length; i += 1) {
    const confidence = clamp01(confidenceMask[i] ?? rawAlphaMask[i] ?? 0);
    const rawValue = rawAlphaMask[i] ?? 0;
    const rawWeight = 0.46 + confidence * 0.16;
    const closedWeight = 0.26 - confidence * 0.05;
    const blurWeight = 1 - rawWeight - closedWeight;
    const preserved = rawValue * (0.88 + confidence * 0.10);
    const refined = rawValue * rawWeight + closed[i] * closedWeight + blurred[i] * blurWeight;
    output[i] = clamp01(Math.max(refined, preserved));
  }

  return output;
}

export class MaskProcessor {
  private previousAlphaMask: Float32Array | null = null;

  process(result: SegmentationFrameResult, tuning?: Pick<VirtualBackgroundTuning, 'confidenceBoost'>): ProcessedMask {
    const { width, height, branches } = result;
    const pixelCount = width * height;
    const rawAlphaMask = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount);
    const confidenceBoost = Math.max(0.25, tuning?.confidenceBoost ?? 1);

    for (const branch of branches) {
      if (branch.kind !== 'human') continue;

      const sourceConfidence = branch.confidenceMask;
      for (let i = 0; i < pixelCount; i += 1) {
        const confidence = sourceConfidence ? clamp01(sourceConfidence[i] ?? 0) : ((branch.categoryMask[i] ?? 0) !== 0 ? 1 : 0);
        const boostedConfidence = sourceConfidence ? clamp01(Math.pow(confidence, 1 / confidenceBoost)) : confidence;
        rawAlphaMask[i] = Math.max(rawAlphaMask[i], boostedConfidence);
        confidenceMask[i] = Math.max(confidenceMask[i], boostedConfidence);
      }
    }

    const alphaMask = refineAlphaMask(rawAlphaMask, confidenceMask, width, height);

    let motionMagnitude = 0;
    const previousAlphaMask = this.previousAlphaMask;
    if (previousAlphaMask && previousAlphaMask.length === pixelCount) {
      let diffSum = 0;
      let activeDiffSum = 0;
      let activePixels = 0;
      for (let i = 0; i < pixelCount; i += 1) {
        const current = alphaMask[i];
        const previous = previousAlphaMask[i];
        const diff = Math.abs(current - previous);
        diffSum += diff;
        if (current > 0.05 || previous > 0.05) {
          activeDiffSum += diff;
          activePixels += 1;
        }
      }
      motionMagnitude = activePixels > 0 ? activeDiffSum / activePixels : diffSum / pixelCount;
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
