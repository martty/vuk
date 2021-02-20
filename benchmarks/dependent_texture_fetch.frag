#version 450
#pragma shader_stage(fragment)

const float offset[5] = float[](0.0, 1.0, 2.0, 3.0, 4.0);
const float weight[5] = float[](0.2270270270, 0.1945945946, 0.1216216216,
                                  0.0540540541, 0.0162162162);

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 FragmentColor;
layout(binding = 0) uniform sampler2D image;

void main() {
    vec2 size = textureSize(image, 0);
    
    FragmentColor = texture(image, uv) * weight[0];
    for (int i=1; i<5; i++) {
        FragmentColor +=
            texture(image, (uv * size + vec2(0.0, offset[i])) / size)
                * weight[i];
        FragmentColor +=
            texture(image, (uv * size - vec2(0.0, offset[i])) / size)
                * weight[i];
    }
}