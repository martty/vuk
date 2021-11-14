#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec3 inColor;

layout (constant_id = 0) const float col_r = 1.0;
layout (constant_id = 1) const float col_g = 1.0;
layout (constant_id = 2) const float col_b = 1.0;

layout (location = 0) out vec4 outFragColor;

void main() {
  vec3 tint = vec3(col_r, col_g, col_b);
  outFragColor = vec4(tint * inColor * sqrt(1 - gl_FragCoord.z), 1.0);
}