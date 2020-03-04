#version 450
#pragma shader_stage(vertex)

out gl_PerVertex 
{
    vec4 gl_Position;
};

layout (location = 0) out vec2 uv;

void main() {
	if(gl_VertexIndex == 0){
		gl_Position = vec4(-1, -1, 0.0, 1.0);
		uv = vec2(0, 2);
	} else if (gl_VertexIndex == 1){
		gl_Position = vec4(-1, 3, 0.0, 1.0);
		uv = vec2(0, 0);
	} else {
		gl_Position = vec4(3, -1, 0.0, 1.0);
		uv = vec2(2, 2);
	}
}
