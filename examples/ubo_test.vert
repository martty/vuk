#version 450
#pragma shader_stage(vertex)
#include "ubo.glsl"

layout(location = 0) in vec3 ipos;

out gl_PerVertex 
{
    vec4 gl_Position;
};

layout (location = 0) out vec3 color;

void main() {
	color = vec3(1,1,1);
    gl_Position = projection * view * model * vec4(ipos, 1.0);
}
