#version 300 es
precision highp float;

uniform sampler2D u_mask;
uniform vec2 u_texelSize;
in vec2 v_uv;
out vec4 outColor;

void main() {
  float sum = 0.0;
  for (int y = -1; y <= 1; y += 1) {
    for (int x = -1; x <= 1; x += 1) {
      vec2 offset = vec2(float(x), float(y)) * u_texelSize;
      sum += texture(u_mask, v_uv + offset).r;
    }
  }
  float blurred = sum / 9.0;
  outColor = vec4(blurred, blurred, blurred, 1.0);
}
