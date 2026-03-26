<template>
  <section class="panel">
    <header class="panel-header">
      <div>
        <p class="eyebrow">Tuning</p>
        <h2>Quality controls</h2>
      </div>
      <span class="badge" :data-state="status === 'running' ? 'live' : 'idle'">
        {{ status === 'running' ? 'Live' : status }}
      </span>
    </header>

    <section class="mini-overlay">
      <div>
        <span>FPS</span>
        <strong>{{ stats.fps.toFixed(1) }}</strong>
      </div>
      <div>
        <span>Latency</span>
        <strong>{{ stats.latencyMs.toFixed(1) }} ms</strong>
      </div>
      <div>
        <span>Seg</span>
        <strong>{{ stats.segmentationMs.toFixed(1) }} ms</strong>
      </div>
      <div>
        <span>Render</span>
        <strong>{{ stats.renderMs.toFixed(1) }} ms</strong>
      </div>
      <div>
        <span>Motion</span>
        <strong>{{ stats.motion.toFixed(2) }}</strong>
      </div>
      <div>
        <span>Light</span>
        <strong>{{ stats.brightness.toFixed(1) }}</strong>
      </div>
      <div>
        <span>Dropped</span>
        <strong>{{ stats.droppedFrames }}</strong>
      </div>
      <div>
        <span>Res</span>
        <strong>{{ stats.processingWidth }}x{{ stats.processingHeight }}</strong>
      </div>
    </section>

    <section class="debug-strip">
      <div>
        <span>Mask fill</span>
        <strong>{{ (stats.foregroundRatio * 100).toFixed(1) }}%</strong>
      </div>
      <div>
        <span>Mask mean</span>
        <strong>{{ stats.maskMean.toFixed(2) }}</strong>
      </div>
      <div>
        <span>Confidence</span>
        <strong>{{ stats.confidenceMean.toFixed(2) }}</strong>
      </div>
      <div>
        <span>Background</span>
        <strong>{{ status }}</strong>
      </div>
    </section>

    <div v-for="control in controls" :key="control.key" class="field">
      <label :for="control.key">
        <span>{{ control.label }}</span>
        <strong>{{ displayValue(control.key) }}</strong>
      </label>
      <input
        :id="control.key"
        :type="control.type"
        :min="control.min"
        :max="control.max"
        :step="control.step"
        :value="String(control.value)"
        @input="update(control.key, ($event.target as HTMLInputElement).value)"
      />
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { EngineStats, VirtualBackgroundTuning } from '@/types/engine';

const props = defineProps<{
  modelValue: VirtualBackgroundTuning;
  status: string;
  stats: EngineStats;
}>();

const emit = defineEmits<{
  (event: 'update:modelValue', value: VirtualBackgroundTuning): void;
}>();

const controls = computed(() => [
  { key: 'temporalAlpha', label: 'Temporal alpha', min: 0.75, max: 0.9, step: 0.01, type: 'range', value: props.modelValue.temporalAlpha },
  { key: 'bilateralSigmaSpatial', label: 'Bilateral sigma spatial', min: 1, max: 8, step: 0.1, type: 'range', value: props.modelValue.bilateralSigmaSpatial },
  { key: 'bilateralSigmaColor', label: 'Bilateral sigma color', min: 0.05, max: 0.4, step: 0.01, type: 'range', value: props.modelValue.bilateralSigmaColor },
  { key: 'feather', label: 'Feather', min: 0.02, max: 0.2, step: 0.005, type: 'range', value: props.modelValue.feather },
  { key: 'lightWrap', label: 'Light wrap', min: 0, max: 0.4, step: 0.01, type: 'range', value: props.modelValue.lightWrap },
  { key: 'confidenceBoost', label: 'Confidence boost', min: 0.8, max: 2, step: 0.05, type: 'range', value: props.modelValue.confidenceBoost },
  { key: 'motionBoost', label: 'Motion boost', min: 1, max: 2, step: 0.05, type: 'range', value: props.modelValue.motionBoost },
  { key: 'brightnessBoost', label: 'Brightness boost', min: 1, max: 2, step: 0.05, type: 'range', value: props.modelValue.brightnessBoost }
]);

function update(key: string, rawValue: string) {
  emit('update:modelValue', {
    ...props.modelValue,
    [key]: Number(rawValue)
  } as VirtualBackgroundTuning);
}

function displayValue(key: string) {
  const value = props.modelValue[key as keyof VirtualBackgroundTuning];
  return typeof value === 'number' ? value.toFixed(2) : String(value);
}
</script>
