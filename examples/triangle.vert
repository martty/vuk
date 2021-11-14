#version 450
#pragma shader_stage(vertex)

out gl_PerVertex 
{
    vec4 gl_Position;
};

layout (location = 0) out vec3 color;

void main() {
	if(gl_VertexIndex == 0){
		gl_Position = vec4(0.0, -0.3, 0.0, 1.0);
		color = vec3(0, 1, 0);
	} else if (gl_VertexIndex == 1){
		gl_Position = vec4(-0.3, 0.3, 0.0, 1.0);
		color = vec3(1, 0, 0);
	} else {
		gl_Position = vec4(0.3, 0.3, 0.0, 1.0);
		color = vec3(0, 0, 1);
	}
}
