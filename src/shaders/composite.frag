#version 300 es
precision highp float;

uniform sampler2D u_person;
uniform sampler2D u_background;
uniform sampler2D u_mask;
uniform sampler2D u_confidence;
uniform vec2 u_texelSize;
uniform float u_feather;
uniform float u_lightWrap;
in vec2 v_uv;
out vec4 outColor;

float sampleSignal(ivec2 coord) {
  ivec2 size = textureSize(u_mask, 0);
  ivec2 clamped = clamp(coord, ivec2(0), size - 1);
  float maskValue = texelFetch(u_mask, clamped, 0).r;
  float confidenceValue = texelFetch(u_confidence, clamped, 0).r;
  return max(maskValue, confidenceValue);
}

void main() {
  ivec2 size = textureSize(u_mask, 0);
  ivec2 coord = clamp(ivec2(gl_FragCoord.xy), ivec2(0), size - 1);
  vec3 fg = texture(u_person, v_uv).rgb;
  vec3 bg = texture(u_background, v_uv).rgb;
  float maskValue = texelFetch(u_mask, coord, 0).r;
  float alpha = smoothstep(u_feather, 1.0 - u_feather, maskValue);
  vec3 light = vec3(u_lightWrap) * (1.0 - alpha) * fg;

  vec3 composite = mix(bg, fg, alpha) + light;
  float centerSignal = sampleSignal(coord);
  float maxSignal = centerSignal;
  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      if (x == 0 && y == 0) continue;
      float signal = sampleSignal(coord + ivec2(x, y));
      maxSignal = max(maxSignal, signal);
    }
  }

  float edge = (1.0 - step(0.35, centerSignal)) * step(0.35, maxSignal);
  composite = mix(composite, vec3(1.0, 0.0, 0.0), edge);
  outColor = vec4(composite, 1.0);
}
