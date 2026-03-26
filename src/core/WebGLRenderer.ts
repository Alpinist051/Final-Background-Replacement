import type { BackgroundMode, BackgroundSource, VirtualBackgroundTuning } from '@/types/engine';
import temporalShaderSource from '@/shaders/temporal.frag?raw';
import bilateralShaderSource from '@/shaders/bilateral.frag?raw';
import compositeShaderSource from '@/shaders/composite.frag?raw';
import blurShaderSource from '@/shaders/blur.frag?raw';

const vertexShaderSource = `#version 300 es
layout(location = 0) in vec2 a_position;
out vec2 v_uv;
void main() {
  vec2 uv = (a_position + 1.0) * 0.5;
  v_uv = uv;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

function parseHexColor(value: string) {
  const hex = value.replace('#', '');
  const normalized =
    hex.length === 3 ? hex.split('').map((character) => character + character).join('') : hex.padEnd(6, '0');
  const red = Number.parseInt(normalized.slice(0, 2), 16) / 255;
  const green = Number.parseInt(normalized.slice(2, 4), 16) / 255;
  const blue = Number.parseInt(normalized.slice(4, 6), 16) / 255;
  return [red, green, blue, 1] as const;
}

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
  const bytes =
    data instanceof Uint8Array
      ? data
      : new Uint8Array(Array.from(data, (value) => Math.max(0, Math.min(255, Math.round(value * 255)))));
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
  backgroundFrame?: ImageBitmap | null;
  background?: BackgroundSource;
  tuning: VirtualBackgroundTuning;
}

export class WebGLRenderer {
  private readonly canvas: OffscreenCanvas;
  private readonly gl: WebGL2RenderingContext | null;
  private readonly fallback2d: OffscreenCanvasRenderingContext2D | null;
  private readonly temporalProgram: WebGLProgram | null;
  private readonly bilateralProgram: WebGLProgram | null;
  private readonly compositeProgram: WebGLProgram | null;
  private readonly blurProgram: WebGLProgram | null;
  private vao: WebGLVertexArrayObject | null = null;
  private framebuffer: WebGLFramebuffer | null = null;
  private width = 1;
  private height = 1;
  private sourceTexture: WebGLTexture | null = null;
  private backgroundTexture: WebGLTexture | null = null;
  private blurTexture: WebGLTexture | null = null;
  private currentMaskTexture: WebGLTexture | null = null;
  private confidenceTexture: WebGLTexture | null = null;
  private previousMaskTexture: WebGLTexture | null = null;
  private temporalTexture: WebGLTexture | null = null;
  private finalMaskTexture: WebGLTexture | null = null;
  private fallbackCanvas: OffscreenCanvas | null = null;
  private fallbackContext2d: OffscreenCanvasRenderingContext2D | null = null;
  private background: BackgroundSource = { mode: 'solid', color: '#111827' };
  private backgroundMode: BackgroundMode = 'solid';

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
      this.blurProgram = createProgram(gl, blurShaderSource);
      this.framebuffer = gl.createFramebuffer();
      this.initializeGeometry();
      this.allocateTextures(1, 1);
    } else {
      this.temporalProgram = null;
      this.bilateralProgram = null;
      this.compositeProgram = null;
      this.blurProgram = null;
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
    this.blurTexture = createTexture(gl, width, height);
    this.currentMaskTexture = createTexture(gl, width, height, gl.R8);
    this.confidenceTexture = createTexture(gl, width, height, gl.R8);
    this.previousMaskTexture = createTexture(gl, width, height, gl.R8);
    this.temporalTexture = createTexture(gl, width, height, gl.R8);
    this.finalMaskTexture = createTexture(gl, width, height, gl.R8);
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

  setBackground(background: BackgroundSource) {
    this.background = background;
    this.backgroundMode = background.mode;
    if (!this.gl || !this.backgroundTexture) return;

    if (background.mode === 'solid') {
      this.applySolidBackground(background.color);
    }
  }

  setBackgroundBitmap(bitmap: ImageBitmap) {
    if (!this.gl || !this.backgroundTexture) return;
    if (this.backgroundMode !== 'image' && this.backgroundMode !== 'video') {
      return;
    }
    uploadBitmap(this.gl, this.backgroundTexture, bitmap);
  }

  async renderFrame(args: RenderFrameArgs) {
    if (!this.gl || !this.temporalProgram || !this.bilateralProgram || !this.compositeProgram || !this.blurProgram) {
      this.renderFallback(args);
      return;
    }

    const { frame, alphaMask, confidenceMask, backgroundFrame, tuning } = args;
    const gl = this.gl;
    this.resize(frame.width, frame.height);

    if (!this.sourceTexture || !this.backgroundTexture || !this.blurTexture || !this.currentMaskTexture || !this.confidenceTexture || !this.previousMaskTexture || !this.temporalTexture || !this.finalMaskTexture) {
      return;
    }

    uploadBitmap(gl, this.sourceTexture, frame);
    uploadMask(gl, this.currentMaskTexture, alphaMask, frame.width, frame.height);
    uploadMask(gl, this.confidenceTexture, confidenceMask, frame.width, frame.height);

    switch (this.backgroundMode) {
      case 'solid':
        this.renderSolidFrame(tuning);
        break;
      case 'image':
        this.renderImageFrame(backgroundFrame, tuning);
        break;
      case 'video':
        this.renderVideoFrame(backgroundFrame, tuning);
        break;
      case 'blur':
        this.renderBlurFrame(tuning);
        break;
    }
    this.swapMaskTextures();
  }

  private applySolidBackground(color: string) {
    if (!this.gl || !this.backgroundTexture) return;
    const rgba = parseHexColor(color);
    const bytes = new Uint8Array(rgba.map((value) => Math.round(value * 255)));
    this.gl.bindTexture(this.gl.TEXTURE_2D, this.backgroundTexture);
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, 1, 1, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, bytes);
  }

  private renderWithMaskAndComposite(backgroundTexture: WebGLTexture | null, tuning: VirtualBackgroundTuning) {
    this.runTemporalPass(tuning);
    this.runBilateralPass(tuning);
    this.runCompositePass(backgroundTexture, tuning);
  }

  private renderSolidFrame(tuning: VirtualBackgroundTuning) {
    this.renderWithMaskAndComposite(this.backgroundTexture, tuning);
  }

  private renderImageFrame(backgroundFrame: ImageBitmap | null | undefined, tuning: VirtualBackgroundTuning) {
    if (backgroundFrame && this.gl && this.backgroundTexture) {
      uploadBitmap(this.gl, this.backgroundTexture, backgroundFrame);
    }
    this.renderWithMaskAndComposite(this.backgroundTexture, tuning);
  }

  private renderVideoFrame(backgroundFrame: ImageBitmap | null | undefined, tuning: VirtualBackgroundTuning) {
    if (backgroundFrame && this.gl && this.backgroundTexture) {
      uploadBitmap(this.gl, this.backgroundTexture, backgroundFrame);
    }
    this.renderWithMaskAndComposite(this.backgroundTexture, tuning);
  }

  private renderBlurFrame(tuning: VirtualBackgroundTuning) {
    this.runBlurPass();
    this.renderWithMaskAndComposite(this.blurTexture, tuning);
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
    if (!this.gl || !this.temporalProgram || !this.temporalTexture) return;
    this.drawToTexture(this.temporalTexture, this.temporalProgram, () => {
      this.bindQuad(this.temporalProgram as WebGLProgram);
      this.setTexture(this.temporalProgram as WebGLProgram, 'u_prevMask', this.previousMaskTexture, 0);
      this.setTexture(this.temporalProgram as WebGLProgram, 'u_currentMask', this.currentMaskTexture, 1);
      this.setTexture(this.temporalProgram as WebGLProgram, 'u_confidence', this.confidenceTexture, 2);
      this.setFloat(this.temporalProgram as WebGLProgram, 'u_alpha', tuning.temporalAlpha);
      this.setFloat(this.temporalProgram as WebGLProgram, 'u_confidenceBoost', tuning.confidenceBoost);
    });
  }

  private runBilateralPass(tuning: VirtualBackgroundTuning) {
    if (!this.gl || !this.bilateralProgram || !this.temporalTexture || !this.finalMaskTexture) return;
    this.drawToTexture(this.finalMaskTexture, this.bilateralProgram, () => {
      this.bindQuad(this.bilateralProgram as WebGLProgram);
      this.setTexture(this.bilateralProgram as WebGLProgram, 'u_mask', this.temporalTexture, 0);
      this.setTexture(this.bilateralProgram as WebGLProgram, 'u_guide', this.sourceTexture, 1);
      this.setVec2(this.bilateralProgram as WebGLProgram, 'u_texelSize', 1 / this.width, 1 / this.height);
      this.setFloat(this.bilateralProgram as WebGLProgram, 'u_sigmaSpatial', tuning.bilateralSigmaSpatial);
      this.setFloat(this.bilateralProgram as WebGLProgram, 'u_sigmaColor', tuning.bilateralSigmaColor);
    });
  }

  private runBlurPass() {
    if (!this.gl || !this.blurProgram || !this.blurTexture || !this.sourceTexture) return;
    this.drawToTexture(this.blurTexture, this.blurProgram, () => {
      this.bindQuad(this.blurProgram as WebGLProgram);
      this.setTexture(this.blurProgram as WebGLProgram, 'u_image', this.sourceTexture, 0);
      this.setVec2(this.blurProgram as WebGLProgram, 'u_texelSize', 1 / this.width, 1 / this.height);
    });
  }

  private runCompositePass(backgroundTexture: WebGLTexture | null, tuning: VirtualBackgroundTuning) {
    if (!this.gl || !this.compositeProgram || !this.sourceTexture || !this.finalMaskTexture || !backgroundTexture) return;
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    this.bindQuad(this.compositeProgram);
    this.setTexture(this.compositeProgram, 'u_person', this.sourceTexture, 0);
    this.setTexture(this.compositeProgram, 'u_background', backgroundTexture, 1);
    this.setTexture(this.compositeProgram, 'u_mask', this.finalMaskTexture, 2);
    this.setVec2(this.compositeProgram, 'u_texelSize', 1 / this.width, 1 / this.height);
    this.setFloat(this.compositeProgram, 'u_feather', tuning.feather);
    this.setFloat(this.compositeProgram, 'u_lightWrap', tuning.lightWrap);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private swapMaskTextures() {
    const nextPrevious = this.previousMaskTexture;
    this.previousMaskTexture = this.finalMaskTexture;
    this.finalMaskTexture = nextPrevious;
  }

  private renderFallback(args: RenderFrameArgs) {
    if (!this.fallback2d) return;
    const { frame, alphaMask, backgroundFrame, tuning } = args;
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

    const background = this.background;

    if (background.mode === 'solid') {
      context.fillStyle = background.color;
      context.fillRect(0, 0, frame.width, frame.height);
    } else if (this.backgroundMode === 'image' || this.backgroundMode === 'video') {
      if (backgroundFrame) {
        context.drawImage(backgroundFrame, 0, 0, frame.width, frame.height);
      } else {
        context.drawImage(frame, 0, 0, frame.width, frame.height);
      }
    } else {
      context.filter = `blur(${Math.max(2, tuning.bilateralSigmaSpatial)}px)`;
      context.drawImage(frame, 0, 0, frame.width, frame.height);
      context.filter = 'none';
    }

    tempContext.drawImage(frame, 0, 0, frame.width, frame.height);
    const imageData = tempContext.getImageData(0, 0, frame.width, frame.height);
    const pixels = imageData.data;
    for (let i = 0; i < alphaMask.length; i += 1) {
      pixels[i * 4 + 3] = Math.round(Math.max(0, Math.min(1, alphaMask[i])) * 255);
    }
    tempContext.putImageData(imageData, 0, 0);
    context.drawImage(this.fallbackCanvas, 0, 0);
  }

  destroy() {
    this.gl?.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
