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
    try {
      return await createImageBitmap(blob, { imageOrientation: 'from-image' });
    } catch (blobError) {
      console.warn('createImageBitmap(blob) failed, retrying via HTMLImageElement.', blobError);
    }
  } catch (err) {
    console.warn('loadImageBitmap fetch path failed, retrying via HTMLImageElement.', err);
  }

  if (typeof Image === 'undefined') {
    throw new Error('Background image decoding is not supported in this environment.');
  }

  const image = new Image();
  image.decoding = 'async';
  image.crossOrigin = 'anonymous';

  const loadPromise = new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error(`Failed to decode background image: ${source}`));
  });

  image.src = source;

  try {
    if ('decode' in image) {
      await image.decode();
    } else {
      await loadPromise;
    }
  } catch {
    await loadPromise;
  }

  try {
    return await createImageBitmap(image, { imageOrientation: 'from-image' });
  } catch (bitmapError) {
    console.warn('createImageBitmap(image) failed, drawing through a canvas fallback.', bitmapError);
  }

  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth || image.width || 1;
  canvas.height = image.naturalHeight || image.height || 1;
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error('Unable to create a background fallback canvas.');
  }

  context.drawImage(image, 0, 0, canvas.width, canvas.height);
  return await createImageBitmap(canvas);
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
