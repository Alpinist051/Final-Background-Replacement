import { defineConfig } from 'vite';
import raw from 'vite-raw-plugin';
import vue from '@vitejs/plugin-vue';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig({
  plugins: [vue(), raw({ fileRegex: /\.(glsl|frag)$/ })],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  worker: { format: 'es' }
});
