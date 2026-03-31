#version 300 es
precision highp float;

uniform sampler2D u_person;
uniform sampler2D u_background;
uniform sampler2D u_mask;
uniform sampler2D u_confidence;
uniform float u_feather;
uniform float u_lightWrap;
in vec2 v_uv;
out vec4 outColor;

void main() {
  vec3 fg = texture(u_person, v_uv).rgb;
  vec3 bg = texture(u_background, v_uv).rgb;
  float maskValue = texture(u_mask, v_uv).r;
  float confidence = texture(u_confidence, v_uv).r;
  float edgeFeather = mix(max(u_feather * 0.20, 0.008), max(u_feather * 0.45, 0.03), smoothstep(0.20, 0.85, confidence));
  float alpha = smoothstep(0.5 - edgeFeather, 0.5 + edgeFeather, maskValue);
  vec3 light = vec3(u_lightWrap) * (1.0 - alpha) * fg;
  outColor = vec4(mix(bg, fg, alpha) + light, 1.0);
}
