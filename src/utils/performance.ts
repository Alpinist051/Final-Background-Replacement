export interface PerformanceSample {
  fps: number;
  latencyMs: number;
  segmentationMs: number;
  renderMs: number;
  brightness: number;
  motion: number;
  droppedFrames: number;
  processingWidth: number;
  processingHeight: number;
}

export function createRollingAverage(size = 30) {
  const values: number[] = [];
  return {
    push(value: number) {
      values.push(value);
      if (values.length > size) values.shift();
    },
    value(): number {
      if (!values.length) return 0;
      return values.reduce((sum, v) => sum + v, 0) / values.length;
    }
  };
}

export function createPerformanceTracker() {
  const fpsAvg = createRollingAverage(30);
  const latencyAvg = createRollingAverage(30);
  const segmentationAvg = createRollingAverage(30);
  const renderAvg = createRollingAverage(30);
  const brightnessAvg = createRollingAverage(30);
  const motionAvg = createRollingAverage(30);

  let droppedFrames = 0;
  let processingWidth = 1280;
  let processingHeight = 720;

  return {
    record(sample: PerformanceSample) {
      fpsAvg.push(sample.fps);
      latencyAvg.push(sample.latencyMs);
      segmentationAvg.push(sample.segmentationMs);
      renderAvg.push(sample.renderMs);
      brightnessAvg.push(sample.brightness);
      motionAvg.push(sample.motion);
      droppedFrames += sample.droppedFrames;
      processingWidth = sample.processingWidth;
      processingHeight = sample.processingHeight;
    },

    snapshot(): PerformanceSample {
      return {
        fps: fpsAvg.value(),
        latencyMs: latencyAvg.value(),
        segmentationMs: segmentationAvg.value(),
        renderMs: renderAvg.value(),
        brightness: brightnessAvg.value(),
        motion: motionAvg.value(),
        droppedFrames,
        processingWidth,
        processingHeight
      };
    },

    resetDroppedFrames() {
      droppedFrames = 0;
    }
  };
}