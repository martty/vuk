#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 ipos;
layout(location = 1) in vec3 icol;

layout(binding = 0) uniform VP {
	mat4 view;
	mat4 projection;
};

out gl_PerVertex 
{
    vec4 gl_Position;
};

layout (location = 0) out vec3 color;

void main() {
	color = icol;
    gl_Position = projection * view * vec4(ipos, 1.0);
}
