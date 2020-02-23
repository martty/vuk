#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 ipos;
//layout(location = 1) in vec3 icolor;

out gl_PerVertex 
{
    vec4 gl_Position;
};

layout (location = 0) out vec3 color;

void main() {
	color = vec3(1,1,1);
    gl_Position = vec4(ipos, 1.0);
}
