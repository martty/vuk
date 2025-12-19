#version 450
#pragma shader_stage(tesscontrol)

layout(vertices = 3) out;

in gl_PerVertex {
    vec4 gl_Position;
} gl_in[gl_MaxPatchVertices];

out gl_PerVertex {
    vec4 gl_Position;
} gl_out[];

layout (location = 0) in vec3 color[];
layout (location = 0) out vec3 colorOut[];

void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    colorOut[gl_InvocationID] = color[gl_InvocationID];
    
    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = 4.0;
        gl_TessLevelOuter[1] = 4.0;
        gl_TessLevelOuter[2] = 4.0;
        
        gl_TessLevelInner[0] = 4.0;
    }
}
