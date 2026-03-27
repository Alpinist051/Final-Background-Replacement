#version 300 es
precision highp float;

uniform sampler2D u_prevMask;
uniform sampler2D u_currentMask;
uniform sampler2D u_confidence;
uniform float u_alpha;
uniform float u_confidenceBoost;
uniform float u_motionBoost;
in vec2 v_uv;
out vec4 outColor;

void main() {
  float prevValue = texture(u_prevMask, v_uv).r;
  float currentValue = texture(u_currentMask, v_uv).r;
  float confidenceValue = clamp(texture(u_confidence, v_uv).r * u_confidenceBoost, 0.0, 1.0);
  float immediateMask = clamp(currentValue * confidenceValue, 0.0, 1.0);
  float motionFactor = clamp(u_motionBoost, 0.0, 1.0);
  float hold = mix(u_alpha, 0.05, motionFactor);
  float carry = prevValue * hold * mix(0.55, 1.0, confidenceValue);
  float temporalMask = max(immediateMask, carry);
  outColor = vec4(temporalMask, temporalMask, temporalMask, 1.0);
}
