#version 460
#pragma shader_stage(vertex)

layout(location = 0) in vec3 ipos;
layout(location = 1) in vec2 iuv;


layout(binding = 0) uniform VP {
	mat4 view;
	mat4 projection;
};

layout(binding = 1) buffer readonly Model {
	mat4 model[];
};

out gl_PerVertex 
{
    vec4 gl_Position;
};

layout (location = 0) precise out vec2 oUV;

void main() {
	oUV = iuv;
    gl_Position = projection * view * model[gl_BaseInstance] * vec4(ipos, 1.0);
}
