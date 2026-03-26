export function isOffscreenCanvas(
  value: CanvasImageSource | OffscreenCanvas | null | undefined
): value is OffscreenCanvas {
  return typeof OffscreenCanvas !== 'undefined' && value instanceof OffscreenCanvas;
}

export function resizeCanvas(
  canvas: HTMLCanvasElement | OffscreenCanvas,
  width: number,
  height: number
) {
  const w = Math.max(1, Math.floor(width));
  const h = Math.max(1, Math.floor(height));
  if (canvas.width === w && canvas.height === h) return;
  canvas.width = w;
  canvas.height = h;
}

export async function loadImageBitmap(source: string): Promise<ImageBitmap> {
  try {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`Failed to load background: ${response.statusText}`);
    const blob = await response.blob();
    return await createImageBitmap(blob, { imageOrientation: 'from-image' });
  } catch (err) {
    console.error('loadImageBitmap failed:', err);
    throw err;
  }
}

export function createAnalysisCanvas(width = 32, height = 18) {
  const canvas = typeof OffscreenCanvas !== 'undefined'
    ? new OffscreenCanvas(width, height)
    : document.createElement('canvas');

  canvas.width = width;
  canvas.height = height;

  const context = canvas.getContext('2d', {
    willReadFrequently: true,
    alpha: false
  })!;

  return { canvas, context };
}