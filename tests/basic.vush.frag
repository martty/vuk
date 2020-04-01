#version 460
#pragma stage(fragment)
#extension GL_GOOGLE_cpp_style_line_directive : require

struct VS_IN {
	vec3 position;
	vec3 normal;
	vec2 texcoord;
};

struct VS_OUT {
	vec4 position;
	vec2 texcoord;
};



struct FS_OUT {
	vec4 color_out;
};

layout(location = 0) out vec4 color_out;

layout(location = 0) in VS_OUT _in;

FS_OUT opaque_fragment(VS_OUT vin) {
	FS_OUT fout;
	fout.color_out = vec4(1,0,0,1);
	return fout;
}

void main(){
	FS_OUT fout = opaque_fragment(_in);
	color_out = fout.color_out;
}