#version 300 es
precision highp float;

uniform sampler2D u_image;
uniform vec2 u_texelSize;
in vec2 v_uv;
out vec4 outColor;

void main() {
  vec3 color = vec3(0.0);
  float total = 0.0;
  for (int y = -4; y <= 4; y++) {
    for (int x = -4; x <= 4; x++) {
      float weight = 1.0 - (length(vec2(float(x), float(y))) / 6.0);
      weight = max(weight, 0.0);
      vec2 uv = v_uv + vec2(float(x), float(y)) * u_texelSize;
      color += texture(u_image, uv).rgb * weight;
      total += weight;
    }
  }
  outColor = vec4(color / max(total, 0.0001), 1.0);
}
