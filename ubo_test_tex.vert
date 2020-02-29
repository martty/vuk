#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 ipos;
layout(location = 1) in vec2 iuv;

layout(binding = 0) uniform VP {
	mat4 view;
	mat4 projection;
};

layout(binding = 1) uniform Model {
	mat4 model;
};

out gl_PerVertex 
{
    vec4 gl_Position;
};

layout (location = 0) out vec3 color;
layout (location = 1) out vec2 oUV;

void main() {
	color = vec3(1,1,1);
	oUV = iuv;
    gl_Position = projection * view * model * vec4(ipos, 1.0);
}
