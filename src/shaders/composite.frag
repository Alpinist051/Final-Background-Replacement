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

  float center = step(0.5, maskValue);
  float left = step(0.5, texture(u_mask, v_uv + vec2(-u_texelSize.x, 0.0)).r);
  float right = step(0.5, texture(u_mask, v_uv + vec2(u_texelSize.x, 0.0)).r);
  float up = step(0.5, texture(u_mask, v_uv + vec2(0.0, u_texelSize.y)).r);
  float down = step(0.5, texture(u_mask, v_uv + vec2(0.0, -u_texelSize.y)).r);
  float edge = max(
    max(abs(center - left), abs(center - right)),
    max(abs(center - up), abs(center - down))
  );

  vec3 composite = mix(bg, fg, alpha) + light;
  composite = mix(composite, vec3(1.0, 0.0, 0.0), clamp(edge, 0.0, 1.0) * 0.9);
  outColor = vec4(composite, 1.0);
}
