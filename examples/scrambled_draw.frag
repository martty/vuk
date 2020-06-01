#version 450 core
#pragma shader_stage(fragment)

layout(location = 0) out vec4 fColor;

layout(binding = 0) uniform sampler2D sTexture;
layout(std430, binding = 1) buffer readonly scramble {
	uint indices[];
};

layout(location = 0) in vec2 UV;

void main() {
	uvec2 size = textureSize(sTexture, 0);
	uvec2 coords = uvec2(size * UV.st);
	uint remap = indices[coords.x + coords.y * size.x];
	ivec2 newcoords = ivec2(remap % size.x, remap / size.y);
    fColor = vec4(texelFetch(sTexture, newcoords, 0).rgb, 1);
}
