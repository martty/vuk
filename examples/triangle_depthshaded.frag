#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;

void main() 
{
  outFragColor = vec4(inColor * sqrt(1 - gl_FragCoord.z), 1.0);
}