import type { VirtualBackgroundTuning } from '@/types/engine';
import temporalShaderSource from '@/shaders/temporal.frag?raw';
import bilateralShaderSource from '@/shaders/bilateral.frag?raw';
import compositeShaderSource from '@/shaders/composite.frag?raw';

const vertexShaderSource = `#version 300 es
layout(location = 0) in vec2 a_position;
out vec2 v_uv;
void main() {
  vec2 uv = (a_position + 1.0) * 0.5;
  v_uv = uv;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

function compileShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new Error('Unable to create shader.');
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader) ?? 'Unknown shader error';
    gl.deleteShader(shader);
    throw new Error(info);
  }
  return shader;
}

function createProgram(gl: WebGL2RenderingContext, fragmentSource: string) {
  const program = gl.createProgram();
  if (!program) {
    throw new Error('Unable to create program.');
  }

  const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  gl.deleteShader(vertexShader);
  gl.deleteShader(fragmentShader);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program) ?? 'Unknown program link error';
    gl.deleteProgram(program);
    throw new Error(info);
  }

  return program;
}

function createTexture(gl: WebGL2RenderingContext, width: number, height: number, internalFormat: number = gl.RGBA8) {
  const texture = gl.createTexture();
  if (!texture) {
    throw new Error('Unable to create texture.');
  }

  const isMaskTexture = internalFormat === gl.R8;
  const format = isMaskTexture ? gl.RED : gl.RGBA;
  const type = gl.UNSIGNED_BYTE;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
  return texture;
}

function uploadBitmap(gl: WebGL2RenderingContext, texture: WebGLTexture, bitmap: ImageBitmap) {
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bitmap);
}

function uploadMask(gl: WebGL2RenderingContext, texture: WebGLTexture, data: Float32Array | Uint8Array, width: number, height: number) {
  const bytes = data instanceof Uint8Array ? data : (() => {
    const output = new Uint8Array(data.length);
    for (let i = 0; i < data.length; i += 1) {
      output[i] = Math.max(0, Math.min(255, Math.round(data[i] * 255)));
    }
    return output;
  })();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, bytes);
}

function getUniform(gl: WebGL2RenderingContext, program: WebGLProgram, name: string) {
  return gl.getUniformLocation(program, name);
}

export interface RenderFrameArgs {
  frame: ImageBitmap;
  alphaMask: Float32Array;
  confidenceMask: Float32Array;
  tuning: VirtualBackgroundTuning;
}

export class WebGLRenderer {
  private readonly canvas: OffscreenCanvas;
  private readonly gl: WebGL2RenderingContext | null;
  private readonly fallback2d: OffscreenCanvasRenderingContext2D | null;
  private readonly temporalProgram: WebGLProgram | null;
  private readonly bilateralProgram: WebGLProgram | null;
  private readonly compositeProgram: WebGLProgram | null;
  private vao: WebGLVertexArrayObject | null = null;
  private framebuffer: WebGLFramebuffer | null = null;
  private width = 1;
  private height = 1;
  private sourceTexture: WebGLTexture | null = null;
  private backgroundTexture: WebGLTexture | null = null;
  private backgroundBitmap: ImageBitmap | null = null;
  private currentMaskTexture: WebGLTexture | null = null;
  private confidenceTexture: WebGLTexture | null = null;
  private previousMaskTexture: WebGLTexture | null = null;
  private temporalTexture: WebGLTexture | null = null;
  private finalMaskTexture: WebGLTexture | null = null;
  private zeroMaskBuffer: Uint8Array | null = null;
  private fallbackCanvas: OffscreenCanvas | null = null;
  private fallbackContext2d: OffscreenCanvasRenderingContext2D | null = null;

  constructor(canvas: OffscreenCanvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      premultipliedAlpha: false
    });
    this.fallback2d = this.gl ? null : canvas.getContext('2d');

    if (this.gl) {
      const gl = this.gl;
      gl.getExtension('EXT_color_buffer_float');
      gl.getExtension('OES_texture_float_linear');

      this.temporalProgram = createProgram(gl, temporalShaderSource);
      this.bilateralProgram = createProgram(gl, bilateralShaderSource);
      this.compositeProgram = createProgram(gl, compositeShaderSource);
      this.framebuffer = gl.createFramebuffer();
      this.initializeGeometry();
      this.allocateTextures(1, 1);
    } else {
      this.temporalProgram = null;
      this.bilateralProgram = null;
      this.compositeProgram = null;
    }
  }

  private initializeGeometry() {
    if (!this.gl) return;
    const gl = this.gl;
    const vao = gl.createVertexArray();
    const buffer = gl.createBuffer();
    if (!vao || !buffer) {
      throw new Error('Unable to initialize WebGL geometry.');
    }

    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);

    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.vao = vao;
  }

  private allocateTextures(width: number, height: number) {
    if (!this.gl) return;
    const gl = this.gl;
    this.width = width;
    this.height = height;

    this.sourceTexture = createTexture(gl, width, height);
    this.backgroundTexture = createTexture(gl, width, height);
    this.currentMaskTexture = createTexture(gl, width, height, gl.R8);
    this.confidenceTexture = createTexture(gl, width, height, gl.R8);
    this.previousMaskTexture = createTexture(gl, width, height, gl.R8);
    this.temporalTexture = createTexture(gl, width, height, gl.R8);
    this.finalMaskTexture = createTexture(gl, width, height, gl.R8);
    this.reapplyBackgroundBitmap();
    gl.viewport(0, 0, width, height);
  }

  resize(width: number, height: number) {
    const nextWidth = Math.max(1, Math.floor(width));
    const nextHeight = Math.max(1, Math.floor(height));
    if (nextWidth === this.width && nextHeight === this.height) return;
    this.canvas.width = nextWidth;
    this.canvas.height = nextHeight;
    this.allocateTextures(nextWidth, nextHeight);
  }

  setBackgroundBitmap(bitmap: ImageBitmap) {
    this.backgroundBitmap?.close();
    this.backgroundBitmap = bitmap;
    this.reapplyBackgroundBitmap();
  }

  async renderFrame(args: RenderFrameArgs) {
    if (!this.gl || !this.compositeProgram) {
      this.renderFallback(args);
      return;
    }

    const { frame, alphaMask, confidenceMask, tuning } = args;
    const gl = this.gl;
    this.resize(frame.width, frame.height);

    if (!this.sourceTexture || !this.backgroundTexture || !this.currentMaskTexture) {
      return;
    }

    uploadBitmap(gl, this.sourceTexture, frame);
    uploadMask(gl, this.currentMaskTexture, alphaMask, frame.width, frame.height);

    this.renderImageFrame(tuning);
  }

  private renderWithMaskAndComposite(backgroundTexture: WebGLTexture | null, _tuning: VirtualBackgroundTuning) {
    this.runCompositePass(backgroundTexture);
  }

  private renderImageFrame(tuning: VirtualBackgroundTuning) {
    this.renderWithMaskAndComposite(this.backgroundTexture, tuning);
  }

  private bindQuad(program: WebGLProgram) {
    if (!this.gl || !this.vao) return;
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.useProgram(program);
  }

  private setTexture(program: WebGLProgram, name: string, texture: WebGLTexture | null, unit: number) {
    if (!this.gl || !texture) return;
    const gl = this.gl;
    const location = getUniform(gl, program, name);
    if (location) {
      gl.uniform1i(location, unit);
    }
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
  }

  private setFloat(program: WebGLProgram, name: string, value: number) {
    if (!this.gl) return;
    const location = getUniform(this.gl, program, name);
    if (location) this.gl.uniform1f(location, value);
  }

  private setVec2(program: WebGLProgram, name: string, x: number, y: number) {
    if (!this.gl) return;
    const location = getUniform(this.gl, program, name);
    if (location) this.gl.uniform2f(location, x, y);
  }

  private drawToTexture(target: WebGLTexture, program: WebGLProgram, renderBody: () => void) {
    if (!this.gl || !this.framebuffer) return;
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
    gl.viewport(0, 0, this.width, this.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    renderBody();
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private runTemporalPass(tuning: VirtualBackgroundTuning) {
    if (!this.gl || !this.temporalProgram || !this.temporalTexture || !this.confidenceTexture) return;
    this.drawToTexture(this.temporalTexture, this.temporalProgram, () => {
      this.bindQuad(this.temporalProgram as WebGLProgram);
      this.setTexture(this.temporalProgram as WebGLProgram, 'u_prevMask', this.previousMaskTexture, 0);
      this.setTexture(this.temporalProgram as WebGLProgram, 'u_currentMask', this.currentMaskTexture, 1);
      this.setTexture(this.temporalProgram as WebGLProgram, 'u_currentConfidence', this.confidenceTexture, 2);
      this.setFloat(this.temporalProgram as WebGLProgram, 'u_alpha', tuning.temporalAlpha);
    });
  }

  private runBilateralPass(tuning: VirtualBackgroundTuning) {
    if (!this.gl || !this.bilateralProgram || !this.temporalTexture || !this.finalMaskTexture || !this.sourceTexture || !this.confidenceTexture) return;
    this.drawToTexture(this.finalMaskTexture, this.bilateralProgram, () => {
      this.bindQuad(this.bilateralProgram as WebGLProgram);
      this.setTexture(this.bilateralProgram as WebGLProgram, 'u_mask', this.temporalTexture, 0);
      this.setTexture(this.bilateralProgram as WebGLProgram, 'u_image', this.sourceTexture, 1);
      this.setTexture(this.bilateralProgram as WebGLProgram, 'u_confidence', this.confidenceTexture, 2);
      this.setVec2(this.bilateralProgram as WebGLProgram, 'u_texelSize', 1 / this.width, 1 / this.height);
      this.setFloat(this.bilateralProgram as WebGLProgram, 'u_sigmaSpatial', tuning.bilateralSigmaSpatial);
      this.setFloat(this.bilateralProgram as WebGLProgram, 'u_sigmaColor', tuning.bilateralSigmaColor);
    });
  }

  private runCompositePass(backgroundTexture: WebGLTexture | null) {
    if (!this.gl || !this.compositeProgram || !this.sourceTexture || !this.currentMaskTexture || !backgroundTexture) return;
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    this.bindQuad(this.compositeProgram);
    this.setTexture(this.compositeProgram, 'u_person', this.sourceTexture, 0);
    this.setTexture(this.compositeProgram, 'u_background', backgroundTexture, 1);
    this.setTexture(this.compositeProgram, 'u_mask', this.currentMaskTexture, 2);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private swapMaskTextures() {
    const nextPrevious = this.previousMaskTexture;
    this.previousMaskTexture = this.finalMaskTexture;
    this.finalMaskTexture = nextPrevious;
  }

  resetMaskHistory() {
    if (!this.gl || !this.previousMaskTexture || !this.finalMaskTexture) return;
    const zeroMask = this.getZeroMaskBuffer(this.width * this.height);
    uploadMask(this.gl, this.previousMaskTexture, zeroMask, this.width, this.height);
    uploadMask(this.gl, this.finalMaskTexture, zeroMask, this.width, this.height);
  }

  private getZeroMaskBuffer(length: number) {
    if (!this.zeroMaskBuffer || this.zeroMaskBuffer.length !== length) {
      this.zeroMaskBuffer = new Uint8Array(length);
    } else {
      this.zeroMaskBuffer.fill(0);
    }
    return this.zeroMaskBuffer;
  }

  private renderFallback(args: RenderFrameArgs) {
    if (!this.fallback2d) return;
    const { frame, alphaMask } = args;
    const context = this.fallback2d;
    this.canvas.width = frame.width;
    this.canvas.height = frame.height;

    if (!this.fallbackCanvas || this.fallbackCanvas.width !== frame.width || this.fallbackCanvas.height !== frame.height) {
      this.fallbackCanvas = new OffscreenCanvas(frame.width, frame.height);
      this.fallbackContext2d = this.fallbackCanvas.getContext('2d', { willReadFrequently: true });
    }

    const tempContext = this.fallbackContext2d;
    if (!tempContext) return;

    context.clearRect(0, 0, frame.width, frame.height);
    tempContext.clearRect(0, 0, frame.width, frame.height);

    const backgroundBitmap = this.backgroundBitmap;

    if (backgroundBitmap) {
      context.drawImage(backgroundBitmap, 0, 0, frame.width, frame.height);
    } else {
      context.fillStyle = '#111827';
      context.fillRect(0, 0, frame.width, frame.height);
    }

    tempContext.drawImage(frame, 0, 0, frame.width, frame.height);
    const imageData = tempContext.getImageData(0, 0, frame.width, frame.height);
    const pixels = imageData.data;

    for (let y = 0; y < frame.height; y += 1) {
      const row = y * frame.width;
      for (let x = 0; x < frame.width; x += 1) {
        const index = row + x;
        const pixelIndex = index * 4;
        pixels[pixelIndex + 3] = Math.round(Math.max(0, Math.min(1, alphaMask[index] ?? 0)) * 255);
      }
    }

    tempContext.putImageData(imageData, 0, 0);
    context.drawImage(this.fallbackCanvas, 0, 0);
  }

  destroy() {
    this.backgroundBitmap?.close();
    this.backgroundBitmap = null;
    this.gl?.getExtension('WEBGL_lose_context')?.loseContext();
  }

  private reapplyBackgroundBitmap() {
    if (!this.gl || !this.backgroundTexture || !this.backgroundBitmap) return;
    uploadBitmap(this.gl, this.backgroundTexture, this.backgroundBitmap);
  }
}
