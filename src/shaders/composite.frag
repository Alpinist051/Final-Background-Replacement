#version 300 es
precision highp float;

uniform sampler2D u_person;
uniform sampler2D u_background;
uniform sampler2D u_mask;
in vec2 v_uv;
out vec4 outColor;

void main() {
  vec3 person = texture(u_person, v_uv).rgb;
  vec3 background = texture(u_background, v_uv).rgb;
  background = clamp((background - 0.5) * 1.05 + 0.5, 0.0, 1.0);
  float alpha = clamp(texture(u_mask, v_uv).r, 0.0, 1.0);
  alpha = smoothstep(0.34, 0.74, alpha);
  outColor = vec4(mix(background, person, alpha), 1.0);
}
