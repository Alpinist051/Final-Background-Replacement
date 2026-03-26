<template>
  <section class="overlay">
    <div>
      <span>Status</span>
      <strong>{{ status }}</strong>
    </div>
    <div>
      <span>FPS</span>
      <strong>{{ stats.fps.toFixed(1) }}</strong>
    </div>
    <div>
      <span>Latency</span>
      <strong>{{ stats.latencyMs.toFixed(1) }} ms</strong>
    </div>
    <div>
      <span>Segmentation</span>
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
      <span>Brightness</span>
      <strong>{{ stats.brightness.toFixed(1) }}</strong>
    </div>
  </section>
</template>

<script setup lang="ts">
import type { EngineStats } from '@/types/engine';

defineProps<{
  status: string;
  stats: EngineStats;
}>();
</script>
