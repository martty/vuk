#version 460
#pragma shader_stage(vertex)
#extension GL_AMD_shader_explicit_vertex_parameter : require

layout(location = 0) in vec3 ipos;
layout(location = 1) in vec2 iUV;

out gl_PerVertex 
{
    vec4 gl_Position;
};

layout (location = 0) out flat uint oIDFlat;
layout (location = 1) out uint oID;

void main() {
	oIDFlat = oID = gl_VertexIndex;
	vec2 uv = 2*(iUV - 0.5);
	uv.y *= -1;
    gl_Position = vec4(uv, 0.0, 1.0);
}
