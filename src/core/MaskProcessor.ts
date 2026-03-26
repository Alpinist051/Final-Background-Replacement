import type { SegmentationFrameResult, VirtualBackgroundTuning } from '@/types/engine';

const HAIR = 1;

export interface ProcessedMask {
  alphaMask: Float32Array;
  confidenceMask: Float32Array;
  motionMagnitude: number;
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

export class MaskProcessor {
  private previousCategoryMask: Uint8Array | null = null;

  process(result: SegmentationFrameResult, tuning: VirtualBackgroundTuning): ProcessedMask {
    const { width, height, categoryMask, confidenceMask: rawConfidence } = result;
    const pixelCount = width * height;

    const dilated = dilateHairOnly(categoryMask, width, height);

    const alphaMask = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount, 1.0);

    for (let i = 0; i < pixelCount; i++) {
      const cat = dilated[i];
      alphaMask[i] = cat === 0 ? 0 : 1; // foreground (multi-class aware)
      if (rawConfidence) confidenceMask[i] = rawConfidence[i];
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

    return { alphaMask, confidenceMask, motionMagnitude };
  }

  reset() {
    this.previousCategoryMask = null;
  }
}