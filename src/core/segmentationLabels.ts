export type SemanticLabelClass = 0 | 1 | 2 | 3 | 4 | 5;

export interface SemanticProfile {
  labelClasses: Uint8Array;
  labelAlphaWeights: Float32Array;
  labelConfidenceWeights: Float32Array;
  classRiseWeights: Float32Array;
  classFallWeights: Float32Array;
  classCarryWeights: Float32Array;
}

const SEMANTIC_ALPHA_WEIGHTS = [0, 1.16, 1.08, 1.12, 1.05, 0.98];
const SEMANTIC_CONFIDENCE_WEIGHTS = [0, 1.08, 1.0, 1.04, 0.96, 0.92];
const SEMANTIC_RISE_WEIGHTS = [0, 0.54, 0.50, 0.52, 0.56, 0.60];
const SEMANTIC_FALL_WEIGHTS = [0, 0.16, 0.18, 0.17, 0.20, 0.22];
const SEMANTIC_CARRY_WEIGHTS = [0, 0.86, 0.82, 0.84, 0.80, 0.76];
const SEMANTIC_PRIORITY_WEIGHTS = [0.86, 1.18, 1.08, 1.12, 1.05, 0.98];

function normalizeSemanticLabel(label: string) {
  return label.trim().toLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' ');
}

function isBackgroundSemanticLabel(label: string) {
  return /background/.test(normalizeSemanticLabel(label));
}

function isHairSemanticLabel(label: string) {
  return normalizeSemanticLabel(label) === 'hair';
}

function isBodySkinSemanticLabel(label: string) {
  const normalized = normalizeSemanticLabel(label);
  return normalized === 'body skin' || normalized === 'body' || normalized === 'skin' || normalized === 'torso';
}

function isFaceSkinSemanticLabel(label: string) {
  const normalized = normalizeSemanticLabel(label);
  return normalized === 'face skin' || normalized === 'face';
}

function isClothingSemanticLabel(label: string) {
  const normalized = normalizeSemanticLabel(label);
  return (
    normalized === 'clothes' ||
    normalized === 'clothing' ||
    normalized === 'shirt' ||
    normalized === 'coat' ||
    normalized === 'jacket' ||
    normalized === 'dress' ||
    normalized === 'pants' ||
    normalized === 'skirt' ||
    normalized === 'sleeve'
  );
}

function isAccessorySemanticLabel(label: string) {
  const normalized = normalizeSemanticLabel(label);
  return (
    normalized === 'others' ||
    normalized === 'other' ||
    normalized === 'accessories' ||
    normalized === 'accessory' ||
    normalized === 'glasses' ||
    normalized === 'sunglasses'
  );
}

export function classifySemanticLabel(label: string): SemanticLabelClass {
  if (!label || isBackgroundSemanticLabel(label)) return 0;
  if (isHairSemanticLabel(label)) return 1;
  if (isBodySkinSemanticLabel(label)) return 2;
  if (isFaceSkinSemanticLabel(label)) return 3;
  if (isClothingSemanticLabel(label)) return 4;
  if (isAccessorySemanticLabel(label)) return 5;
  return 5;
}

export function getSemanticLabelPriority(label: string) {
  return SEMANTIC_PRIORITY_WEIGHTS[classifySemanticLabel(label)];
}

export function getBackgroundIndex(labels: string[]) {
  return labels.findIndex((label) => isBackgroundSemanticLabel(label));
}

export function buildSemanticProfile(labels: string[]): SemanticProfile {
  const size = Math.max(1, labels.length);
  const labelClasses = new Uint8Array(size);
  const labelAlphaWeights = new Float32Array(size);
  const labelConfidenceWeights = new Float32Array(size);
  const classRiseWeights = new Float32Array(SEMANTIC_RISE_WEIGHTS);
  const classFallWeights = new Float32Array(SEMANTIC_FALL_WEIGHTS);
  const classCarryWeights = new Float32Array(SEMANTIC_CARRY_WEIGHTS);

  for (let i = 0; i < size; i += 1) {
    const labelClass = classifySemanticLabel(labels[i] ?? '');
    labelClasses[i] = labelClass;
    labelAlphaWeights[i] = SEMANTIC_ALPHA_WEIGHTS[labelClass] ?? SEMANTIC_ALPHA_WEIGHTS[0];
    labelConfidenceWeights[i] = SEMANTIC_CONFIDENCE_WEIGHTS[labelClass] ?? SEMANTIC_CONFIDENCE_WEIGHTS[0];
    classRiseWeights[labelClass] = SEMANTIC_RISE_WEIGHTS[labelClass] ?? SEMANTIC_RISE_WEIGHTS[0];
    classFallWeights[labelClass] = SEMANTIC_FALL_WEIGHTS[labelClass] ?? SEMANTIC_FALL_WEIGHTS[0];
    classCarryWeights[labelClass] = SEMANTIC_CARRY_WEIGHTS[labelClass] ?? SEMANTIC_CARRY_WEIGHTS[0];
  }

  return {
    labelClasses,
    labelAlphaWeights,
    labelConfidenceWeights,
    classRiseWeights,
    classFallWeights,
    classCarryWeights
  };
}
