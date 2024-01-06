#version 450
#pragma shader_stage(vertex)

out gl_PerVertex 
{
    vec4 gl_Position;
	float gl_PointSize;
};

layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 speed;
layout (location = 0) out vec2 ospeed;

void main() {
	ospeed = speed;
	gl_Position = vec4(pos, 0.1, 1);
	gl_PointSize = 5.0;
}
