#version 450
#pragma shader_stage(vertex)

layout(location = 0) in vec3 ipos;
layout(location = 1) in vec3 icol;
layout(location = 2) in vec3 inorm;

layout(binding = 0) uniform VP {
	mat4 view;
	mat4 projection;
};

layout (push_constant) uniform PushConstants {
	vec3 camPos;
    uint use_smooth_normals;
	uint view_space_grid;
};

out gl_PerVertex 
{
    vec4 gl_Position;
};

layout (location = 0) out vec3 color;
layout (location = 1) out vec3 pos;
layout (location = 2) out vec3 norm;

void main() {
	color = icol;
    gl_Position = projection * view * vec4(ipos, 1.0);
	pos = ipos;
	norm = normalize(vec3(inorm.x, inorm.y, inorm.z));
	/*if(view_space_grid == 1){
		norm = -vec3(inverse(view) * vec4(norm, 0));
		gl_Position = projection * view * vec4(ipos, 1.0);
	}*/
}
