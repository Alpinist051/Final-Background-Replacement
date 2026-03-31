<template>
  <main class="app-shell">
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow">Browser-only background replacement</p>
        <h1>Virtual background demo</h1>
        <p class="lede">
          Upload a background image and replace the camera feed in real time.
        </p>
      </div>
      <div class="hero-metrics">
        <div class="metric-card">
          <span>Status</span>
          <strong>{{ state.status }}</strong>
        </div>
        <div class="metric-card">
          <span>Live FPS</span>
          <strong>{{ state.stats.fps.toFixed(1) }}</strong>
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
    </section>
  </main>
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue';
import CameraInput from '@/components/CameraInput.vue';
import BackgroundSelector from '@/components/BackgroundSelector.vue';
import OutputPreview from '@/components/OutputPreview.vue';
import { useVirtualBackground } from '@/composables/useVirtualBackground';
import type { ImageBackground } from '@/types/engine';
import { DEFAULT_BACKGROUND } from '@/constants/defaultBackground';

const { state, attachCanvas, start, stop, setBackground } = useVirtualBackground();
const devices = ref<MediaDeviceInfo[]>([]);
const deviceId = ref('');
const background = ref<ImageBackground>({ ...DEFAULT_BACKGROUND });

async function refreshDevices() {
  const entries = await navigator.mediaDevices.enumerateDevices();
  devices.value = entries.filter((entry) => entry.kind === 'videoinput');
}

watch(background, (value) => { void setBackground(value); }, { immediate: true });

onMounted(() => {
  void refreshDevices();
});
</script>
