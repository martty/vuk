#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 ipos;
layout(location = 1) in vec3 inormal;
layout(location = 2) in vec2 iuv;

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

layout (location = 0) out vec3 position;
layout (location = 1) out vec3 normal;
layout (location = 2) out vec2 uv;

void main() {
	position = vec3(model * vec4(ipos, 1.0));
	uv = iuv;
	normal = vec3(model * vec4(inormal, 0.0));
    gl_Position = projection * view * model * vec4(ipos, 1.0);
}
