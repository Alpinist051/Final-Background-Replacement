#version 300 es
precision highp float;

uniform sampler2D u_person;
uniform sampler2D u_background;
uniform sampler2D u_mask;
uniform vec2 u_texelSize;
uniform float u_feather;
uniform float u_lightWrap;
in vec2 v_uv;
out vec4 outColor;

void main() {
  vec3 fg = texture(u_person, v_uv).rgb;
  vec3 bg = texture(u_background, v_uv).rgb;
  float maskValue = texture(u_mask, v_uv).r;
  float alpha = smoothstep(u_feather, 1.0 - u_feather, maskValue);
  vec3 light = vec3(u_lightWrap) * (1.0 - alpha) * fg;

  vec3 composite = mix(bg, fg, alpha) + light;
  outColor = vec4(composite, 1.0);
}
