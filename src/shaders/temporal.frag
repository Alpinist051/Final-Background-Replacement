#version 300 es
precision highp float;

uniform sampler2D u_prevMask;
uniform sampler2D u_currentMask;
uniform float u_alpha;
in vec2 v_uv;
out vec4 outColor;

void main() {
  float prevValue = texture(u_prevMask, v_uv).r;
  float currentValue = texture(u_currentMask, v_uv).r;
  float maskValue = mix(prevValue, currentValue, clamp(u_alpha, 0.0, 1.0));
  outColor = vec4(maskValue, maskValue, maskValue, 1.0);
}
