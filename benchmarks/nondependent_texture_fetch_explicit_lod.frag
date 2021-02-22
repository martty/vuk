#version 450
#pragma shader_stage(fragment)

const float weight[5] = float[](0.2270270270, 0.1945945946, 0.1216216216,
                                  0.0540540541, 0.0162162162);

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 FragmentColor;
layout(binding = 0) uniform sampler2D image;

void main() {
    FragmentColor = textureLod(image, uv, 0) * weight[0];
    FragmentColor += textureLodOffset(image, uv, 0, ivec2(0, 1)) * weight[1];
    FragmentColor += textureLodOffset(image, uv, 0, ivec2(0,-1)) * weight[1];
    FragmentColor += textureLodOffset(image, uv, 0, ivec2(0, 2)) * weight[2];
    FragmentColor += textureLodOffset(image, uv, 0, ivec2(0,-2)) * weight[2];
    FragmentColor += textureLodOffset(image, uv, 0, ivec2(0, 3)) * weight[3];
    FragmentColor += textureLodOffset(image, uv, 0, ivec2(0,-3)) * weight[3];
    FragmentColor += textureLodOffset(image, uv, 0, ivec2(0, 4)) * weight[4];
    FragmentColor += textureLodOffset(image, uv, 0, ivec2(0,-4)) * weight[4];
   
}