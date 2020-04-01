#version 450
#pragma stage(fragment)

layout(location = 0) in struct VS_OUT {
	vec4 position;
	vec2 texcoord;
} vin;

layout(binding = 0) uniform sampler2D _t1;

struct FS_OUT {
	vec4 color_out;
};

layout(location = 0) out vec4 color_out;

FS_OUT opaque_fragment(VS_OUT vin, sampler2D t1) {
	FS_OUT fout;
	fout.color_out = vec4(texture(t1, vin.texcoord).rgb, 1);
	return fout;
}

void main(){
	FS_OUT fout = opaque_fragment(vin, _t1);
	color_out = fout.color_out;
}