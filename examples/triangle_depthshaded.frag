#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec3 inPos;
layout (location = 2) in vec3 inNorm;

layout (binding = 1) uniform samplerCube env;

layout (push_constant) uniform PushConstants {
    vec3 camPos;
    uint use_smooth_normals;
	uint _;
};

layout (location = 0) out vec4 outFragColor;

vec3 fresnel_schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
	vec3 dx = vec3(dFdx(inPos.x), dFdx(inPos.y), dFdx(inPos.z));
	vec3 dy = vec3(dFdy(inPos.x), dFdy(inPos.y), dFdy(inPos.z));
	vec3 N;
	if(use_smooth_normals == 1)
		N = normalize(inNorm); 
	else 
		N = normalize(cross(dx, dy));
	vec3 V = normalize(camPos - inPos);
	vec3 R = reflect(-V, N);
	R.x *= -1;
	vec3 environ = texture(env, R).xyz;
	vec3 F0 = inColor;
	
	vec3 F = fresnel_schlick(max(dot(N, V), 0.0), F0);
	vec3 hdr_color = environ * inColor;
	vec3 ldr_color = hdr_color / (1+hdr_color);
	outFragColor = vec4(ldr_color, 1.0);//vec4(inColor * sqrt(1 - gl_FragCoord.z), 1.0);
}