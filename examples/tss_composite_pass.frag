#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec2 iUV;

layout(binding = 2) uniform sampler2D screen_tex;

layout (location = 0) out vec4 result;
layout (location = 1) out uvec4 ouv;

void main() {
	vec2 lod_result = textureQueryLod(screen_tex, iUV);
	float clamped_lod = clamp(lod_result.x, 0, 9);
	precise uvec2 uv = uvec2(floor(iUV * textureSize(screen_tex, int(clamped_lod))));
	//ivec2 uv = ivec2(floor(iUV * textureSize(screen_tex, lod)));
	result = texelFetch(screen_tex, ivec2(uv), int(clamped_lod));
	ouv = uvec4(uv, clamped_lod, 0);
}