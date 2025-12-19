#version 450
#pragma shader_stage(tesseval)

layout(triangles, equal_spacing, ccw) in;

in gl_PerVertex {
    vec4 gl_Position;
} gl_in[gl_MaxPatchVertices];

out gl_PerVertex {
    vec4 gl_Position;
};

layout (location = 0) in vec3 colorIn[];
layout (location = 0) out vec3 colorOut;

void main() {
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec4 p2 = gl_in[2].gl_Position;
    
    vec3 tc = gl_TessCoord;
    gl_Position = tc.x * p0 + tc.y * p1 + tc.z * p2;
    
    colorOut = tc.x * colorIn[0] + tc.y * colorIn[1] + tc.z * colorIn[2];
}
