#version 450 core
#pragma shader_stage(fragment)

layout (location = 0) out vec4 gPosition;
layout (location = 1) out vec4 gNormal;
layout (location = 2) out vec4 gAlbedoSpec;

layout (location = 0) in vec3 FragPos;
layout (location = 1) in vec3 Normal;
layout (location = 2) in vec2 TexCoords;

layout (binding = 2) uniform samplerCube env_map;

layout (push_constant) uniform PushConstants {
    vec3 camPos;
};

vec3 fresnel_schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main()
{    
    // store the fragment position vector in the first gbuffer texture
    gPosition = vec4(FragPos, 1);
    // also store the per-fragment normals into the gbuffer
    gNormal = vec4(normalize(Normal), 1);
    // and the diffuse per-fragment color
    vec3 V = normalize(camPos - FragPos);
	vec3 R = reflect(-V, gNormal.xyz);
	R.x *= -1;
	vec3 environ = texture(env_map, R).xyz;
    vec3 inColor = vec3(1,0.5,0.5);
	vec3 F0 = inColor;
	
	vec3 F = fresnel_schlick(max(dot(gNormal.xyz, V), 0.0), F0);
	vec3 hdr_color = environ * inColor;
	vec3 ldr_color = hdr_color / (1+hdr_color);
	gAlbedoSpec = vec4(environ, 1.0);//vec4(inColor * sqrt(1 - gl_FragCoord.z), 1.0);
}  