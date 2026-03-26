import type { SegmentationFrameResult, VirtualBackgroundTuning } from '@/types/engine';

const HAIR = 1;

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

function dilateHairOnly(categoryMask: Uint8Array, width: number, height: number): Uint8Array {
  const output = new Uint8Array(categoryMask);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (categoryMask[i] !== HAIR) continue;
      // 3×3 dilation – only on hair class (preserves fine hair strands)
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = Math.max(0, Math.min(width - 1, x + dx));
          const ny = Math.max(0, Math.min(height - 1, y + dy));
          output[ny * width + nx] = HAIR;
        }
      }
    }
  }
  return output;
}

function dilateForeground(categoryMask: Uint8Array, width: number, height: number): Uint8Array {
  const output = new Uint8Array(categoryMask);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (categoryMask[i] === 0) continue;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = Math.max(0, Math.min(width - 1, x + dx));
          const ny = Math.max(0, Math.min(height - 1, y + dy));
          output[ny * width + nx] = categoryMask[i];
        }
      }
    }
  }
  return output;
}

export class MaskProcessor {
  private previousCategoryMask: Uint8Array | null = null;

  process(result: SegmentationFrameResult, tuning: VirtualBackgroundTuning): ProcessedMask {
    const { width, height, categoryMask, confidenceMask: rawConfidence } = result;
    const pixelCount = width * height;

    const dilatedForeground = dilateForeground(categoryMask, width, height);
    const dilated = dilateHairOnly(dilatedForeground, width, height);

    const alphaMask = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount, 1.0);
    let foregroundPixels = 0;
    let alphaSum = 0;
    let confidenceSum = 0;

    for (let i = 0; i < pixelCount; i++) {
      const cat = dilated[i];
      const confidence = rawConfidence ? Math.max(0, Math.min(1, rawConfidence[i])) : 1;
      if (cat === 0) {
        alphaMask[i] = 0;
      } else {
        // Keep detected foreground opaque so the matte reads clearly,
        // while preserving a tiny amount of soft edge guidance.
        alphaMask[i] = Math.max(0.94, confidence * 0.98);
        foregroundPixels += 1;
      }
      confidenceMask[i] = confidence;
      alphaSum += alphaMask[i];
      confidenceSum += confidence;
    }

    // Precise motion magnitude for fast-motion boost
    let motionMagnitude = 0;
    if (this.previousCategoryMask) {
      let diff = 0;
      for (let i = 0; i < pixelCount; i++) {
        if (dilated[i] !== this.previousCategoryMask[i]) diff++;
      }
      motionMagnitude = diff / pixelCount;
    }

    this.previousCategoryMask = new Uint8Array(dilated);

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
  }
}
