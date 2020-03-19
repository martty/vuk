#version 450 core
#pragma shader_stage(fragment)

layout (location = 0) out vec4 gPosition;
layout (location = 1) out vec4 gNormal;
layout (location = 2) out vec4 gAlbedoSpec;

layout (location = 0) in vec3 FragPos;
layout (location = 1) in vec3 Normal;
layout (location = 2) in vec2 TexCoords;

void main()
{    
    // store the fragment position vector in the first gbuffer texture
    gPosition = vec4(FragPos, 1);
    // also store the per-fragment normals into the gbuffer
    gNormal = vec4(normalize(Normal), 1);
    // and the diffuse per-fragment color
    gAlbedoSpec = vec4(1,0,0.5,1);
}  