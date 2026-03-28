<template>
  <main class="app-shell">
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow">Browser-only background replacement</p>
        <h1>Virtual background engine</h1>
        <p class="lede">
          MediaPipe multi-person semantic segmentation, worker scheduling, and WebGL compositing in a single Vue 3 app.
        </p>
      </div>
      <div class="hero-metrics">
        <div class="metric-card">
          <span>Mode</span>
          <strong>{{ state.status }}</strong>
        </div>
        <div class="metric-card">
          <span>Live FPS</span>
          <strong>{{ state.stats.fps.toFixed(1) }}</strong>
        </div>
        <div class="metric-card">
          <span>Pipeline</span>
          <strong>Worker + GPU</strong>
        </div>
      </div>
    </section>

    <section class="workspace">
      <div class="left-column">
        <OutputPreview label="Live output" @ready="attachCanvas" />
        <div class="control-grid">
          <CameraInput
            :devices="devices"
            :device-id="deviceId"
            :running="state.status === 'running'"
            :error="state.error"
            @update:deviceId="deviceId = $event"
            @refresh="refreshDevices"
            @start="start"
            @stop="stop"
          />
          <BackgroundSelector v-model="background" />
        </div>
      </div>

      <div class="right-column">
        <ControlsPanel v-model="tuning" :status="state.status" :stats="state.stats" />
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue';
import CameraInput from '@/components/CameraInput.vue';
import BackgroundSelector from '@/components/BackgroundSelector.vue';
import ControlsPanel from '@/components/ControlsPanel.vue';
import OutputPreview from '@/components/OutputPreview.vue';
import { useVirtualBackground } from '@/composables/useVirtualBackground';
import type { BackgroundSource, VirtualBackgroundTuning } from '@/types/engine';

const { state, attachCanvas, start, stop, setTuning, setBackground } = useVirtualBackground();
const devices = ref<MediaDeviceInfo[]>([]);
const deviceId = ref('');
const background = ref<BackgroundSource>({ mode: 'solid', color: '#111827' });
const tuning = computed<VirtualBackgroundTuning>({
  get: () => state.tuning,
  set: (value) => setTuning(value)
});

async function refreshDevices() {
  const entries = await navigator.mediaDevices.enumerateDevices();
  devices.value = entries.filter((entry) => entry.kind === 'videoinput');
}

watch(
  background,
  (value) => {
    void setBackground(value);
  },
  { deep: true }
);

onMounted(() => {
  void refreshDevices();
});
</script>
