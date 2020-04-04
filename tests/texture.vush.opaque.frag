#version 450
#pragma stage(fragment)

layout(location = 0) in struct VS_OUT {
	vec4 position;
	vec2 texcoord;
} vin;

struct Texture2D {
    uint Container;
    float Page;
};

layout(binding = 0) uniform sampler2DArray texture_2D_storage[32];

vec4 texture(Texture2D addr, vec2 texcoord){
	vec3 texCoord = vec3(texcoord.xy, addr.Page);
	return texture(texture_2D_storage[addr.Container], texCoord);
}

struct _manual {
	Texture2D t1;
};

layout(std140, binding = 1) uniform _manual_ {
	_manual _;
} _fragment_manual;


struct FS_OUT {
	vec4 color_out;
};

layout(location = 0) out vec4 color_out;

FS_OUT opaque_fragment(VS_OUT vin, Texture2D t1) {
	FS_OUT fout;
	fout.color_out = vec4(texture(t1, vin.texcoord).rgb, 1);
	return fout;
}

void main(){
	FS_OUT fout = opaque_fragment(vin, _fragment_manual._.t1);
	color_out = fout.color_out;
}