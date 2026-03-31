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
        <OutputPreview
          eyebrow="Final output"
          title="Final composited frame"
          label="Live"
          @ready="attachCanvas"
        />
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
        <section class="debug-section">
          <header class="panel-header debug-header">
            <div>
              <p class="eyebrow">Debug</p>
              <h2>Foreground stages</h2>
            </div>
            <span class="badge">Raw / cleaned / final</span>
          </header>
          <div class="debug-grid">
            <OutputPreview
              eyebrow="Stage 1"
              title="Raw MediaPipe mask"
              label="Raw"
              canvas-class="preview-canvas--compact"
              @ready="attachRawMaskCanvas"
            />
            <OutputPreview
              eyebrow="Stage 2"
              title="Cleaned mask"
              label="Refined"
              canvas-class="preview-canvas--compact"
              @ready="attachCleanedMaskCanvas"
            />
          </div>
          <p class="hint debug-hint">
            Raw is the direct MediaPipe output. Cleaned is after our mask cleanup. If raw is wrong, the detection input is the problem. If raw looks right but cleaned shifts or grows, the cleanup is too aggressive. If both masks look right but the final view is off, the composite stage is the one to inspect.
          </p>
        </section>
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

const { state, attachCanvas, attachDebugCanvases, start, stop, setBackground } = useVirtualBackground();
const devices = ref<MediaDeviceInfo[]>([]);
const deviceId = ref('');
const background = ref<ImageBackground>({ ...DEFAULT_BACKGROUND });
const rawMaskCanvas = ref<HTMLCanvasElement | null>(null);
const cleanedMaskCanvas = ref<HTMLCanvasElement | null>(null);

async function refreshDevices() {
  const entries = await navigator.mediaDevices.enumerateDevices();
  devices.value = entries.filter((entry) => entry.kind === 'videoinput');
}

function syncDebugCanvases() {
  if (!rawMaskCanvas.value || !cleanedMaskCanvas.value) return;
  attachDebugCanvases(rawMaskCanvas.value, cleanedMaskCanvas.value);
}

function attachRawMaskCanvas(canvas: HTMLCanvasElement) {
  rawMaskCanvas.value = canvas;
  syncDebugCanvases();
}

function attachCleanedMaskCanvas(canvas: HTMLCanvasElement) {
  cleanedMaskCanvas.value = canvas;
  syncDebugCanvases();
}

watch(background, (value) => { void setBackground(value); }, { immediate: true });

onMounted(() => {
  void refreshDevices();
});
</script>
