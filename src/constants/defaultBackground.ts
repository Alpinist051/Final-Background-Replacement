import type { ImageBackground } from '@/types/engine';

export const DEFAULT_BACKGROUND: ImageBackground = {
  mode: 'image',
  url: new URL('../assets/demo-background.svg', import.meta.url).href,
  label: 'Demo background'
};
