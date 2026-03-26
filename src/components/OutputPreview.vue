<template>
  <section class="panel preview-panel">
    <header class="panel-header">
      <div>
        <p class="eyebrow">Live output</p>
        <h2>Processed preview</h2>
      </div>
      <span class="badge">{{ label }}</span>
    </header>
    <canvas ref="canvasEl" class="preview-canvas"></canvas>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';

defineProps<{
  label: string;
}>();

const emit = defineEmits<{
  (event: 'ready', canvas: HTMLCanvasElement): void;
}>();

const canvasEl = ref<HTMLCanvasElement | null>(null);

onMounted(() => {
  if (canvasEl.value) {
    emit('ready', canvasEl.value);
  }
});
</script>
