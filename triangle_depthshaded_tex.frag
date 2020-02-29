#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout(binding = 2) uniform sampler2D tex; 

void main() 
{
  vec3 col = texture(tex, vec2(inUV.x, 1-inUV.y)).xyz;
  if(any(greaterThan(col, vec3(0.8)))) discard;
  outFragColor = vec4(inColor * pow(1 - gl_FragCoord.z, 1/3) * texture(tex, vec2(inUV.x, 1-inUV.y)).xyz, 1.0);
}