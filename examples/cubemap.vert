#version 450
#pragma shader_stage(vertex)

layout (location = 0) in vec3 in_pos;

layout (location = 0) out vec3 out_pos;

layout (binding = 0) uniform Projection {
    mat4 projection;
};

layout (binding = 1) uniform View {
    mat4 view;
};

void main()
{
    out_pos = in_pos;
    gl_Position =  projection * view * vec4(out_pos, 1.0);
}