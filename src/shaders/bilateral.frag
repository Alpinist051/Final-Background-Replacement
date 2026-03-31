#version 300 es
precision highp float;

uniform sampler2D u_mask;
uniform sampler2D u_image;
uniform sampler2D u_confidence;
uniform vec2 u_texelSize;
uniform float u_sigmaSpatial;
uniform float u_sigmaColor;
in vec2 v_uv;
out vec4 outColor;

float gaussian(float x, float sigma) {
  float safeSigma = max(sigma, 0.0001);
  return exp(-(x * x) / (2.0 * safeSigma * safeSigma));
}

void main() {
  const int RADIUS = 2;
  vec3 centerColor = texture(u_image, v_uv).rgb;
  float centerMask = texture(u_mask, v_uv).r;
  float centerConfidence = texture(u_confidence, v_uv).r;
  float sigmaSpatial = max(u_sigmaSpatial, 0.0001);
  float sigmaColor = max(u_sigmaColor, 0.0001);
  float sum = 0.0;
  float weightSum = 0.0;

  for (int y = -RADIUS; y <= RADIUS; y += 1) {
    for (int x = -RADIUS; x <= RADIUS; x += 1) {
      vec2 offset = vec2(float(x), float(y)) * u_texelSize;
      vec2 sampleUv = clamp(v_uv + offset, vec2(0.0), vec2(1.0));
      float sampleMask = texture(u_mask, sampleUv).r;
      vec3 sampleColor = texture(u_image, sampleUv).rgb;
      float sampleConfidence = texture(u_confidence, sampleUv).r;
      float spatialWeight = gaussian(length(vec2(float(x), float(y))), sigmaSpatial);
      float colorWeight = gaussian(length(sampleColor - centerColor), sigmaColor);
      float confidenceWeight = mix(0.35, 1.0, sampleConfidence);
      float weight = spatialWeight * colorWeight * confidenceWeight;
      sum += sampleMask * weight;
      weightSum += weight;
    }
  }

  float refined = weightSum > 0.0 ? sum / weightSum : centerMask;
  float centerConfidenceEase = smoothstep(0.20, 0.85, centerConfidence);
  float blendAmount = mix(0.74, 0.34, centerConfidenceEase);
  float outputMask = mix(centerMask, refined, blendAmount);
  outputMask = max(outputMask, centerMask * 0.92);
  outColor = vec4(outputMask, outputMask, outputMask, 1.0);
}
