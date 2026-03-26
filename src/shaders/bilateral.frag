#version 300 es
precision highp float;

uniform sampler2D u_mask;
uniform sampler2D u_guide;
uniform vec2 u_texelSize;
uniform float u_sigmaSpatial;
uniform float u_sigmaColor;
in vec2 v_uv;
out vec4 outColor;

float gaussian(float x, float sigma) {
  return exp(-(x * x) / (2.0 * sigma * sigma));
}

void main() {
  vec3 centerGuide = texture(u_guide, v_uv).rgb;
  float centerMask = texture(u_mask, v_uv).r;
  float weightSum = 0.0;
  float maskSum = 0.0;

  for (int y = -4; y <= 4; y++) {
    for (int x = -4; x <= 4; x++) {
      vec2 offset = vec2(float(x), float(y)) * u_texelSize;
      vec2 uv = v_uv + offset;
      vec3 guide = texture(u_guide, uv).rgb;
      float mask = texture(u_mask, uv).r;
      float spatial = gaussian(length(vec2(float(x), float(y))), u_sigmaSpatial);
      float colorDistance = length(centerGuide - guide);
      float color = gaussian(colorDistance, u_sigmaColor);
      float weight = spatial * color;
      weightSum += weight;
      maskSum += mask * weight;
    }
  }

  float blurred = weightSum > 0.0 ? maskSum / weightSum : centerMask;
  outColor = vec4(blurred, blurred, blurred, 1.0);
}
