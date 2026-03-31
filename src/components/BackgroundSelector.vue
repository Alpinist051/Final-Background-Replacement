<template>
  <section class="panel compact-panel">
    <header class="panel-header compact-header">
      <div>
        <p class="eyebrow">Background</p>
        <h2>Background image</h2>
      </div>
      <span class="badge">Built-in demo</span>
    </header>

    <label class="field">
      <span>Upload background image</span>
      <input type="file" accept="image/*" @change="handleFile" />
    </label>

    <p v-if="fileLabel" class="hint">{{ fileLabel }}</p>
    <p v-else class="hint">A demo background is loaded by default. Upload your own image to replace it.</p>
  </section>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue';
import type { ImageBackground } from '@/types/engine';

defineProps<{
  modelValue: ImageBackground;
}>();

const emit = defineEmits<{
  (event: 'update:modelValue', value: ImageBackground): void;
}>();

const fileLabel = ref('');
let objectUrl: string | null = null;

function handleFile(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
  }
  objectUrl = URL.createObjectURL(file);
  fileLabel.value = file.name;
  emit('update:modelValue', { mode: 'image', url: objectUrl, label: file.name });
  input.value = '';
}

onBeforeUnmount(() => {
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
    objectUrl = null;
  }
});
</script>
