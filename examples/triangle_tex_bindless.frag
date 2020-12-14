#version 450
#pragma shader_stage(fragment)
#extension GL_EXT_nonuniform_qualifier : require

layout (location = 1) in vec2 inUV;
layout (location = 0) flat in uint base_instance;

layout (location = 0) out vec4 outFragColor;

layout(set = 1, binding = 0) uniform sampler2D[] tex; 

void main() 
{
  vec3 col = texture(tex[base_instance], vec2(inUV.x, inUV.y)).xyz;
  if(any(greaterThan(col, vec3(0.8)))) discard;
  outFragColor = vec4(col, 1.0);
}