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

type BranchKind = SegmentationFrameResult['branches'][number]['kind'];
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

function dilateBinaryMask(source: Uint8Array, width: number, height: number, radius = 1) {
  const output = new Uint8Array(source.length);
  for (let y = 0; y < height; y += 1) {
    const minY = Math.max(0, y - radius);
    const maxY = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x += 1) {
      const minX = Math.max(0, x - radius);
      const maxX = Math.min(width - 1, x + radius);
      let value = 0;
      for (let sampleY = minY; sampleY <= maxY && value === 0; sampleY += 1) {
        const row = sampleY * width;
        for (let sampleX = minX; sampleX <= maxX; sampleX += 1) {
          if (source[row + sampleX] !== 0) {
            value = 1;
            break;
          }
        }
      }
      output[y * width + x] = value;
    }
  }
  return output;
}

function buildClassProfile(kind: BranchKind, labels: string[]): LabelProfile {
  const size = Math.max(1, labels.length);
  const weights = new Float32Array(size);
  const classes = new Uint8Array(size);

  for (let i = 0; i < size; i += 1) {
    const labelClass = classifyLabel(labels[i] ?? '');
    classes[i] = labelClass;

    if (labelClass === 0) {
      weights[i] = 0;
    } else if (labelClass === 1) {
      weights[i] = kind === 'selfie' ? 1.1 : 0.96;
    } else if (labelClass === 3) {
      weights[i] = kind === 'selfie' ? 1.02 : 1.08;
    } else {
      weights[i] = kind === 'selfie' ? 1.01 : 1.0;
    }
  }

  return { weights, classes };
}

function branchBoost(kind: BranchKind) {
  return kind === 'selfie' ? 1.04 : 1.0;
}

export class MaskProcessor {
  private previousAlphaMask: Float32Array | null = null;
  private classProfileKey = '';
  private classProfile: LabelProfile = {
    weights: new Float32Array([0, 1]),
    classes: new Uint8Array([0, 2])
  };

  private getClassProfile(kind: BranchKind, labels: string[]) {
    const key = `${kind}|${labels.join('|')}`;
    if (key !== this.classProfileKey) {
      this.classProfileKey = key;
      this.classProfile = buildClassProfile(kind, labels);
    }
    return this.classProfile;
  }

  process(result: SegmentationFrameResult, tuning: VirtualBackgroundTuning, liveMotion = 0): ProcessedMask {
    const { width, height, branches } = result;
    const pixelCount = width * height;
    const motionFactor = clamp01(liveMotion * 3.5);
    const hasSelfieBranch = branches.some((branch) => branch.kind === 'selfie');
    const orderedBranches = [
      ...branches.filter((branch) => branch.kind === 'selfie'),
      ...branches.filter((branch) => branch.kind === 'subject')
    ];

    const selfieAlpha = createFloatBuffer(pixelCount);
    const subjectHumanAlpha = createFloatBuffer(pixelCount);
    const subjectObjectAlpha = createFloatBuffer(pixelCount);
    const confidenceMask = createFloatBuffer(pixelCount);
    const tuningBoost = Math.min(1.32, Math.max(0.94, tuning.confidenceBoost));

    for (const branch of orderedBranches) {
      const classProfile = this.getClassProfile(branch.kind, branch.labels);
      const kindBoost = branchBoost(branch.kind);
      const ageLimit = branch.kind === 'subject'
        ? (motionFactor > 0.45 ? 140 : motionFactor > 0.2 ? 220 : 360)
        : 0;
      const ageFloor = branch.kind === 'subject'
        ? (motionFactor > 0.45 ? 0.2 : motionFactor > 0.2 ? 0.34 : 0.5)
        : 1;
      const ageFactor = branch.kind === 'subject' ? Math.max(ageFloor, 1 - branch.ageMs / ageLimit) : 1;
      const sourceConfidence = branch.confidenceMask;

      for (let i = 0; i < pixelCount; i += 1) {
        const categoryIndex = branch.categoryMask[i] ?? 0;
        const labelClass = classProfile.classes[categoryIndex] ?? 0;
        if (labelClass === 0) continue;

        const confidence = sourceConfidence ? clamp01(sourceConfidence[i]) : 1;
        const classWeight = classProfile.weights[categoryIndex] ?? 1;
        let alpha = confidence * classWeight * kindBoost * ageFactor * tuningBoost;

        if (branch.kind === 'selfie') {
          alpha = Math.max(alpha, confidence * 0.95);
          selfieAlpha[i] = Math.max(selfieAlpha[i], alpha);
          confidenceMask[i] = Math.max(confidenceMask[i], clamp01(confidence * ageFactor));
          continue;
        }

        if (labelClass === 1) {
          if (confidence >= 0.45) {
            subjectHumanAlpha[i] = Math.max(subjectHumanAlpha[i], Math.max(alpha * 0.78, confidence * 0.68));
          }
        } else {
          const objectThreshold = labelClass === 3 ? 0.35 : 0.45;
          if (confidence >= objectThreshold) {
            subjectObjectAlpha[i] = Math.max(
              subjectObjectAlpha[i],
              Math.max(alpha * (labelClass === 3 ? 0.98 : 0.9), confidence * (labelClass === 3 ? 0.82 : 0.75))
            );
          }
        }

        confidenceMask[i] = Math.max(confidenceMask[i], clamp01(confidence * ageFactor));
      }
    }

    const selfieSupport = new Uint8Array(pixelCount);
    for (let i = 0; i < pixelCount; i += 1) {
      selfieSupport[i] = selfieAlpha[i] > 0.16 ? 1 : 0;
    }
    const selfieHalo = hasSelfieBranch ? dilateBinaryMask(selfieSupport, width, height, 2) : null;

    const previousAlphaMask = this.previousAlphaMask;
    const nextAlphaMask = createFloatBuffer(pixelCount);

    let motionMagnitude = 0;
    let foregroundPixels = 0;
    let alphaSum = 0;
    let confidenceSum = 0;

    for (let i = 0; i < pixelCount; i += 1) {
      const previousAlpha = previousAlphaMask?.[i] ?? 0;

      let currentAlpha = selfieAlpha[i];

      if (subjectObjectAlpha[i] > currentAlpha) {
        currentAlpha = subjectObjectAlpha[i];
      }

      const allowSubjectHuman = !hasSelfieBranch
        || (selfieHalo?.[i] ?? 0) > 0
        || subjectHumanAlpha[i] >= 0.72
        || previousAlpha >= 0.55;
      if (allowSubjectHuman && subjectHumanAlpha[i] > currentAlpha) {
        currentAlpha = subjectHumanAlpha[i];
      }

      let alpha = clamp01(currentAlpha);
      if (previousAlphaMask) {
        const stability = clamp01(confidenceMask[i] * 0.85 + (1 - motionFactor) * 0.2);
        const riseBlend = 0.42 + stability * 0.28;
        const fallBlend = 0.18 + stability * 0.3;
        alpha = alpha >= previousAlpha
          ? previousAlpha + (alpha - previousAlpha) * riseBlend
          : previousAlpha + (alpha - previousAlpha) * fallBlend;
      }
      alpha = clamp01(alpha);
      nextAlphaMask[i] = alpha;

      if (previousAlphaMask && Math.abs(alpha - previousAlpha) > 0.12) {
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
