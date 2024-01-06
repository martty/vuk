#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec2 speed;
layout (location = 0) out vec4 outFragColor;

void main() {
	vec2 coord = gl_PointCoord - vec2(0.5);
	outFragColor = vec4(10*vec3(length(speed)), 1-smoothstep(0.3, 0.5, length(coord)));
}