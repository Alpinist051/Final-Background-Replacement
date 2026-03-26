#version 300 es
precision highp float;

uniform sampler2D u_prevMask;
uniform sampler2D u_currentMask;
uniform sampler2D u_confidence;
uniform float u_alpha;
uniform float u_confidenceBoost;
in vec2 v_uv;
out vec4 outColor;

void main() {
  float prevValue = texture(u_prevMask, v_uv).r;
  float currentValue = texture(u_currentMask, v_uv).r;
  float confidenceValue = texture(u_confidence, v_uv).r * u_confidenceBoost;
  float blended = mix(prevValue, currentValue, clamp(u_alpha * confidenceValue, 0.0, 1.0));
  outColor = vec4(blended, blended, blended, 1.0);
}
