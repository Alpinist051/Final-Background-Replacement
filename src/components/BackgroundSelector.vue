<template>
  <section class="panel compact-panel">
    <header class="panel-header compact-header">
      <div class="header-line">
        <p class="eyebrow">Scene</p>
        <span class="badge">{{ modelValue.mode }}</span>
      </div>
      <h2>Background</h2>
    </header>

    <div class="inline-controls scene-controls">
      <div class="button-row compact-actions mode-row">
        <button :class="['chip', modelValue.mode === 'solid' ? 'active' : '']" @click="setSolid()">
          Solid
        </button>
        <button :class="['chip', modelValue.mode === 'image' ? 'active' : '']" @click="setImage()">
          Image
        </button>
        <button :class="['chip', modelValue.mode === 'video' ? 'active' : '']" @click="setVideo()">
          Video
        </button>
        <button :class="['chip', modelValue.mode === 'blur' ? 'active' : '']" @click="setBlur()">
          Blur
        </button>
      </div>

      <template v-if="modelValue.mode === 'solid'">
        <label class="inline-field inline-color">
          <span>Color</span>
          <input type="color" :value="solidColor" @input="updateSolid(($event.target as HTMLInputElement).value)" />
        </label>
      </template>

      <template v-if="modelValue.mode === 'image' || modelValue.mode === 'video'">
        <label class="inline-field inline-file">
          <span>{{ modelValue.mode === 'image' ? 'File' : 'File' }}</span>
          <input type="file" :accept="modelValue.mode === 'image' ? 'image/*' : 'video/*'" @change="handleFile" />
        </label>
      </template>

      <template v-if="modelValue.mode === 'blur'">
        <label class="inline-field inline-range">
          <span>Blur</span>
          <input
            type="range"
            min="2"
            max="18"
            step="1"
            :value="modelValue.strength"
            @input="updateBlur(Number(($event.target as HTMLInputElement).value))"
          />
        </label>
      </template>
    </div>

    <p v-if="fileLabel" class="hint compact-hint">{{ fileLabel }}</p>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue';
import type { BackgroundSource } from '@/types/engine';

const props = defineProps<{
  modelValue: BackgroundSource;
}>();

const emit = defineEmits<{
  (event: 'update:modelValue', value: BackgroundSource): void;
}>();

const fileLabel = ref('');
let objectUrl: string | null = null;
const solidColor = computed(() => (props.modelValue.mode === 'solid' ? props.modelValue.color : '#0f172a'));

function setSolid() {
  emit('update:modelValue', {
    mode: 'solid',
    color: props.modelValue.mode === 'solid' ? props.modelValue.color : '#0f172a'
  });
}

function setImage() {
  emit('update:modelValue', {
    mode: 'image',
    url: 'https://images.unsplash.com/photo-1493246507139-91e8fad9978e?auto=format&fit=crop&w=1280&q=80'
  });
  fileLabel.value = 'Preset image';
}

function setVideo() {
  emit('update:modelValue', {
    mode: 'video',
    url: 'https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4',
    loop: true
  });
  fileLabel.value = 'Preset video';
}

function setBlur() {
  emit('update:modelValue', {
    mode: 'blur',
    strength: props.modelValue.mode === 'blur' ? props.modelValue.strength : 8
  });
  fileLabel.value = '';
}

function updateSolid(color: string) {
  emit('update:modelValue', { mode: 'solid', color });
}

function updateBlur(strength: number) {
  emit('update:modelValue', { mode: 'blur', strength });
}

function handleFile(event: Event) {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  if (objectUrl) {
    URL.revokeObjectURL(objectUrl);
  }
  objectUrl = URL.createObjectURL(file);
  fileLabel.value = file.name;
  if (file.type.startsWith('video/')) {
    emit('update:modelValue', { mode: 'video', url: objectUrl, loop: true, label: file.name });
    input.value = '';
    return;
  }
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
