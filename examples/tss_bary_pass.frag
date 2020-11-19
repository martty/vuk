#version 450
#pragma shader_stage(fragment)
#extension GL_AMD_shader_explicit_vertex_parameter : require

layout (location = 0) in flat uint iIDFlat;
layout (location = 1) in __explicitInterpAMD uint iID;

layout (location = 0) out vec2 bary;

void main() { 
	uint id0 = interpolateAtVertexAMD(iID, 0);
	uint id1 = interpolateAtVertexAMD(iID, 1);
	uint id2 = interpolateAtVertexAMD(iID, 2);

	vec3 barycentrics;
	if (iIDFlat == id0)
	{
		barycentrics.y = gl_BaryCoordNoPerspAMD.x;
		barycentrics.z = gl_BaryCoordNoPerspAMD.y;
		barycentrics.x = 1.0 - barycentrics.z - barycentrics.y;
	}
	else if (iIDFlat == id1)
	{
		barycentrics.x = gl_BaryCoordNoPerspAMD.x;
		barycentrics.y = gl_BaryCoordNoPerspAMD.y;
		barycentrics.z = 1.0 - barycentrics.x - barycentrics.y;
	}
	else if (iIDFlat == id2)
	{
		barycentrics.z = gl_BaryCoordNoPerspAMD.x;
		barycentrics.x = gl_BaryCoordNoPerspAMD.y;
		barycentrics.y = 1.0 - barycentrics.x - barycentrics.z;
	}
	else
	{
		barycentrics = vec3(1.0);
	}
	
	//packed_bary = packUnorm2x16(barycentrics.xy);
	//barycentrics.y = 1 - barycentrics.y;
	//barycentrics.x = 1 - barycentrics.x;
	bary = barycentrics.xy;
}