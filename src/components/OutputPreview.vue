<template>
  <section class="panel preview-panel">
    <header class="panel-header">
      <div>
        <p class="eyebrow">{{ eyebrow }}</p>
        <h2>{{ title }}</h2>
      </div>
      <span class="badge">{{ label }}</span>
    </header>
    <canvas ref="canvasEl" :class="['preview-canvas', canvasClass]"></canvas>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';

withDefaults(defineProps<{
  eyebrow?: string;
  title?: string;
  label: string;
  canvasClass?: string;
}>(), {
  eyebrow: 'Live output',
  title: 'Processed preview',
  canvasClass: ''
});

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
