#version 300 es
precision highp float;

uniform sampler2D u_prevMask;
uniform sampler2D u_currentMask;
uniform sampler2D u_currentConfidence;
uniform float u_alpha;
in vec2 v_uv;
out vec4 outColor;

void main() {
  float prevValue = texture(u_prevMask, v_uv).r;
  float currentValue = texture(u_currentMask, v_uv).r;
  float confidence = texture(u_currentConfidence, v_uv).r;
  float confidenceHold = 1.0 - smoothstep(0.25, 0.80, confidence);
  float currentWeight = clamp(u_alpha - confidenceHold * 0.12, 0.0, 1.0);
  float maskValue = mix(prevValue, currentValue, currentWeight);
  outColor = vec4(maskValue, maskValue, maskValue, 1.0);
}
