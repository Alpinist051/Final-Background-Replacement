<template>
  <section class="panel compact-panel">
    <header class="panel-header compact-header">
      <div class="header-line">
        <p class="eyebrow">Capture</p>
        <span class="badge" :data-state="running ? 'live' : 'idle'">
          {{ running ? 'Live' : 'Idle' }}
        </span>
      </div>
      <h2>Camera input</h2>
    </header>

    <div class="inline-controls">
      <label class="inline-field inline-select">
        <span>Device</span>
        <select :value="deviceId" @change="emit('update:deviceId', ($event.target as HTMLSelectElement).value)">
          <option value="">Default camera</option>
          <option v-for="device in devices" :key="device.deviceId" :value="device.deviceId">
            {{ device.label || `Camera ${device.deviceId.slice(0, 6)}` }}
          </option>
        </select>
      </label>

      <div class="button-row compact-actions">
        <button class="primary" :disabled="running" @click="emit('start')">Start</button>
        <button class="secondary" :disabled="!running" @click="emit('stop')">Stop</button>
        <button class="ghost" @click="emit('refresh')">Refresh</button>
      </div>
    </div>

    <p v-if="error" class="error compact-error">{{ error }}</p>
  </section>
</template>

<script setup lang="ts">
defineProps<{
  devices: MediaDeviceInfo[];
  deviceId: string;
  running: boolean;
  error: string | null;
}>();

const emit = defineEmits<{
  (event: 'start'): void;
  (event: 'stop'): void;
  (event: 'refresh'): void;
  (event: 'update:deviceId', deviceId: string): void;
}>();
</script>
