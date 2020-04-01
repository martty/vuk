#version 460
#pragma stage(vertex)

#define _BASEINSTANCE_SIZE 32
#define _DRAW_SIZE 32
#define _MATERIAL_SIZE 32
#define _MATERIAL_INSTANCE_SIZE 32

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

struct _baseinstance {
	mat4 model_matrix;
};

layout(std140, binding = 1) uniform _baseinstance_ {
	_baseinstance _[_BASEINSTANCE_SIZE];
} _vertex_baseinstance;

layout(push_constant) uniform _Mat {
    uint _material_index;
} _push_contants;

struct Material {
  vec4 some_color;
  float size;
};

struct _material {
	float size;
};

struct _material_instance {
	vec4 some_color;
};

layout(std140, binding = 2) uniform _material_ {
	_material _[_MATERIAL_SIZE];
} _vertex_material;

layout(std140, binding = 3) uniform _material_instance_ {
	_material_instance _[_MATERIAL_INSTANCE_SIZE];
} _vertex_material_instance;

layout(location = 0) out struct VS_OUT {
	vec4 position;
	vec4 color;
	vec2 texcoord;
} vout;

VS_OUT opaque_vertex(VS_IN vin, VP vp, mat4 model_matrix, Material material) {
	VS_OUT vout;
	vout.position = vp.projection * vp.view * model_matrix * vec4(vin.position * material.size, 1.0);
	vout.texcoord = vin.texcoord;
	vout.color = material.some_color;
	return vout;
}

void main(){
	VS_IN vin;
	vin.position = _opaque_vertex_position;
	vin.normal = _opaque_vertex_normal;
	vin.texcoord = _opaque_vertex_texcoord;
	Material material;
	material.some_color = _vertex_material_instance._[gl_BaseInstance].some_color;
	material.size = _vertex_material._[_push_contants._material_index].size;
	vout = opaque_vertex(vin, _vertex_vp._, _vertex_baseinstance._[gl_BaseInstance].model_matrix, material);
}