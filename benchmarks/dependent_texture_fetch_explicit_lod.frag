#version 450
#pragma shader_stage(fragment)

const float offset[5] = float[](0.0 / 112, 1.0 / 112, 2.0 / 112, 3.0 / 112, 4.0 / 112);
const float weight[5] = float[](0.2270270270, 0.1945945946, 0.1216216216,
                                  0.0540540541, 0.0162162162);

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 FragmentColor;
layout(binding = 0) uniform sampler2D image;

layout(push_constant) uniform PC {
    float image_size_rcp;
};

void main() {
    FragmentColor = textureLod(image, uv, 0) * weight[0];
	float offset = 0;
    for (int i=1; i<5; i++) {
		offset += image_size_rcp;
        FragmentColor += textureLod(image, uv + vec2(0.0, offset), 0) * weight[i];
        FragmentColor += textureLod(image, uv - vec2(0.0, offset), 0) * weight[i];
    }
}