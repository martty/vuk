#version 450
#pragma shader_stage(fragment)

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 FragmentColor;
layout(binding = 0) uniform sampler2D image;

void main() {
    FragmentColor = textureLod(image, uv, 0);
}