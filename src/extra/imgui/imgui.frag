#version 450 core
#pragma shader_stage(fragment)

layout(location = 0) out vec4 fColor;

layout(set=0, binding=0) uniform sampler2D sTexture;

layout(location = 0) in struct {
    vec4 Color;
    vec2 UV;
} In;

layout(push_constant) uniform uPushConstant {
    vec2 uScale;
    vec2 uTranslate;
    int uIsSrgb;
} pc;

vec3 srgb_to_linear(vec3 color) {
    return mix(
        pow(abs((color + 0.055) / 1.055), vec3(2.4)),
        color / 12.92,
        lessThan(color, vec3(0.04045))
    );
}

void main()
{
    vec4 sampled_color = texture(sTexture, In.UV.st) * In.Color;
    if (pc.uIsSrgb == 0) {
        sampled_color.rgb = srgb_to_linear(sampled_color.rgb);
    }

    fColor = sampled_color;
}
