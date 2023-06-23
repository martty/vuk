#version 450
#pragma shader_stage(vertex)
#include "ubo.glsl"

layout(location = 0) in vec3 ipos;

out gl_PerVertex 
{
    vec4 gl_Position;
};

layout (location = 0) out vec3 color;

#ifdef SCALE
const float scale = SCALE;
#else
const float scale = 1.0;
#endif

void main() {
	color = vec3(1,1,1);
    gl_Position = projection * view * model * vec4(scale * ipos, 1.0);
}
