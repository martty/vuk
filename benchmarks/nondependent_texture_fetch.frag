#version 450
#pragma shader_stage(fragment)

const float weight[5] = float[](0.2270270270, 0.1945945946, 0.1216216216,
                                  0.0540540541, 0.0162162162);

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 FragmentColor;
layout(binding = 0) uniform sampler2D image;

void main() {
    vec2 size = textureSize(image, 0);
    
    FragmentColor = texture(image, uv) * weight[0];
   
    FragmentColor += textureOffset(image, uv, ivec2(0, 1)) * weight[1];
    FragmentColor += textureOffset(image, uv, ivec2(0,-1)) * weight[1];
    FragmentColor += textureOffset(image, uv, ivec2(0, 2)) * weight[2];
    FragmentColor += textureOffset(image, uv, ivec2(0,-2)) * weight[2];
    FragmentColor += textureOffset(image, uv, ivec2(0, 3)) * weight[3];
    FragmentColor += textureOffset(image, uv, ivec2(0,-3)) * weight[3];
    FragmentColor += textureOffset(image, uv, ivec2(0, 4)) * weight[4];
    FragmentColor += textureOffset(image, uv, ivec2(0,-4)) * weight[4];
   
}