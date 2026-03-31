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
  float confidenceEase = smoothstep(0.22, 0.85, confidence);
  float feather = clamp(mix(u_feather * 1.9, u_feather * 1.0, confidenceEase), 0.025, 0.14);
  float alpha = smoothstep(0.5 - feather, 0.5 + feather, maskValue);
  float edgeSoftness = (1.0 - confidenceEase) * pow(1.0 - alpha, 1.2);
  vec3 edgeBg = mix(bg, fg, edgeSoftness * 0.08);
  vec3 light = vec3(u_lightWrap) * edgeSoftness * fg * (1.0 - confidenceEase * 0.25);
  outColor = vec4(mix(edgeBg, fg, alpha) + light, 1.0);
}
