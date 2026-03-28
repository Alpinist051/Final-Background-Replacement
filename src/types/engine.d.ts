export type BackgroundMode = 'image' | 'video' | 'solid' | 'blur';

export interface SolidBackground {
  mode: 'solid';
  color: string;
}

export interface ImageBackground {
  mode: 'image';
  url: string;
  label?: string;
}

export interface VideoBackground {
  mode: 'video';
  url: string;
  label?: string;
  loop?: boolean;
}

export interface BlurBackground {
  mode: 'blur';
  strength?: number; // optional - used only for UI
}

export type BackgroundSource = SolidBackground | ImageBackground | VideoBackground | BlurBackground;

export interface VirtualBackgroundTuning {
  temporalAlpha: number;
  bilateralSigmaSpatial: number;
  bilateralSigmaColor: number;
  feather: number;
  lightWrap: number;
  confidenceBoost: number;
  motionBoost: number;
  brightnessBoost: number;
}

export interface EngineStats {
  fps: number;
  latencyMs: number;
  segmentationMs: number;
  renderMs: number;
  brightness: number;
  motion: number;
  droppedFrames: number;
  processingWidth: number;
  processingHeight: number;
  foregroundRatio: number;
  maskMean: number;
  confidenceMean: number;
}

export interface EngineState {
  status: 'idle' | 'starting' | 'running' | 'stopping' | 'error';
  error: string | null;
  stats: EngineStats;
  tuning: VirtualBackgroundTuning;
  background: BackgroundSource;
}

export interface SegmentationFrameResult {
  width: number;
  height: number;
  branches: SegmentationBranchResult[];
}

export type SegmentationBranchKind = 'human';

export interface SegmentationBranchResult {
  kind: SegmentationBranchKind;
  width: number;
  height: number;
  categoryMask: Uint8Array;
  confidenceMask?: Float32Array;
  labels: string[];
  ageMs: number;
}

export interface QualityUpdate {
  width: number;
  height: number;
  temporalAlpha: number;
}
