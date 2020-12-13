#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec3 inPos;
layout (location = 2) in vec3 inNorm;

layout (binding = 1) uniform samplerCube env;

layout (location = 0) out vec4 outFragColor;

void main() {
	vec3 dx = vec3(dFdx(inPos.x), dFdx(inPos.y), dFdx(inPos.z));
	vec3 dy = vec3(dFdy(inPos.x), dFdy(inPos.y), dFdy(inPos.z));
	vec3 N = normalize(inNorm); //normalize(cross(-dx, dy));
	vec3 environ = texture(env, N).xyz;
	vec3 hdr_color = environ;
	vec3 ldr_color = environ / (1+environ);
	outFragColor = vec4(ldr_color, 1.0);//vec4(inColor * sqrt(1 - gl_FragCoord.z), 1.0);
}