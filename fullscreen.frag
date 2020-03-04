#version 450 core
#pragma shader_stage(fragment)

layout(location = 0) out vec4 fColor;

layout(set=0, binding=0) uniform sampler2D sTexture;

layout(location = 0) in vec2 UV;

void main()
{
    fColor = vec4(texture(sTexture, UV.st).xxx, 1);
}
