#version 460
#pragma stage(vertex)

#define _BASEINSTANCE_SIZE 32
#define _DRAW_SIZE 32

layout(location = 0) in vec3 _opaque_vertex_position;
layout(location = 1) in vec3 _opaque_vertex_normal;
layout(location = 2) in vec2 _opaque_vertex_texcoord;

struct VS_IN {
	vec3 position;
	vec3 normal;
	vec2 texcoord;
};

struct VP {
	mat4 view;
	mat4 projection;
};

layout(std140, binding = 0) uniform _pass_ {
	VP _;
} _vertex_vp;

struct _manual {
	vec4 tint;
};

layout(std140, binding = 1) uniform _manual_ {
	_manual _;
} _vertex_tint;

struct _baseinstance {
	mat4 model_matrix;
};

layout(std140, binding = 2) uniform _baseinstance_ {
	_baseinstance _[_BASEINSTANCE_SIZE];
} _vertex_baseinstance;

struct _draw {
	vec4 color;
};

layout(std140, binding = 3) uniform _draw_ {
	_draw _[_DRAW_SIZE];
} _vertex_color;

layout(location = 0) out struct VS_OUT {
	vec4 position;
	vec4 color;
	vec2 texcoord;
} vout;


VS_OUT opaque_vertex(VS_IN vin, VP vp, vec4 tint, mat4 model_matrix, vec4 color) {
	VS_OUT vout;
	vout.position = vp.projection * vp.view * model_matrix * vec4(vin.position, 1.0);
	vout.texcoord = vin.texcoord;
	vout.color = color * tint;
	return vout;
}

void main(){
	VS_IN vin;
	vin.position = _opaque_vertex_position;
	vin.normal = _opaque_vertex_normal;
	vin.texcoord = _opaque_vertex_texcoord;
	vout = opaque_vertex(vin, _vertex_vp._, _vertex_tint._.tint, _vertex_baseinstance._[gl_BaseInstance].model_matrix, _vertex_color._[gl_DrawID].color);
}