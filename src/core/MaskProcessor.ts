import type { SegmentationFrameResult } from '@/types/engine';

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

function cleanBinaryMask(source: Uint8Array, width: number, height: number) {
  const output = new Uint8Array(source.length);

  for (let y = 0; y < height; y += 1) {
    const top = Math.max(0, y - 1);
    const bottom = Math.min(height - 1, y + 1);

    for (let x = 0; x < width; x += 1) {
      const left = Math.max(0, x - 1);
      const right = Math.min(width - 1, x + 1);
      let foreground = 0;
      let sampleCount = 0;

      for (let sampleY = top; sampleY <= bottom; sampleY += 1) {
        const row = sampleY * width;
        for (let sampleX = left; sampleX <= right; sampleX += 1) {
          sampleCount += 1;
          if ((source[row + sampleX] ?? 0) !== 0) {
            foreground += 1;
          }
        }
      }

      output[y * width + x] = foreground >= Math.ceil(sampleCount / 2) ? 1 : 0;
    }
  }

  return output;
}

export class MaskProcessor {
  private previousAlphaMask: Float32Array | null = null;

  process(result: SegmentationFrameResult): ProcessedMask {
    const { width, height, branches } = result;
    const pixelCount = width * height;
    const alphaMask = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount);

    for (const branch of branches) {
      if (branch.kind !== 'human') continue;

      const sourceConfidence = branch.confidenceMask;
      const threshold = sourceConfidence ? 0.6 : 0.5;
      const binaryMask = new Uint8Array(branch.categoryMask.length);

      for (let i = 0; i < pixelCount; i += 1) {
        const confidence = sourceConfidence ? Math.max(0, Math.min(1, sourceConfidence[i] ?? 0)) : (branch.categoryMask[i] ?? 0);
        confidenceMask[i] = Math.max(confidenceMask[i], confidence);
        binaryMask[i] = confidence >= threshold ? 1 : 0;
      }

      const cleanedMask = cleanBinaryMask(binaryMask, branch.width, branch.height);
      for (let i = 0; i < pixelCount; i += 1) {
        alphaMask[i] = cleanedMask[i] ? 1 : alphaMask[i];
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
