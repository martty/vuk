#version 450
#pragma stage(fragment)

layout(location = 0) in struct VS_OUT {
	vec4 position;
	vec2 texcoord;
} vin;

layout(binding = 0) uniform sampler wrap_linear;
layout(binding = 1) uniform texture2D _t1_texture;

vec4 texture(sampler samp, texture2D t2D, vec2 texcoord){
	return texture(sampler2D(t2D, samp), texcoord);
}

struct FS_OUT {
	vec4 color_out;
};

layout(location = 0) out vec4 color_out;

FS_OUT opaque_fragment(VS_OUT vin, texture2D t1) {
	FS_OUT fout;
	fout.color_out = vec4(texture(wrap_linear, t1, vin.texcoord).rgb, 1);
	return fout;
}

void main(){
	FS_OUT fout = opaque_fragment(vin, _t1_texture);
	color_out = fout.color_out;
}