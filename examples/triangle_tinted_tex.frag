#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

layout(binding = 2) uniform sampler2D tex; 
layout(binding = 3) uniform Material {
	vec4 tint; 
};

void main() 
{
  vec3 col = texture(tex, vec2(inUV.x, 1-inUV.y)).xyz;
  if(any(greaterThan(col, vec3(0.8)))) discard;
  vec3 tinted = col + (tint.xyz - col) * tint.a;
  outFragColor = vec4(inColor * tinted, 1.0);
}