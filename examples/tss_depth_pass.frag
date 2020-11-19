#version 450
#pragma shader_stage(fragment)
#extension GL_AMD_shader_explicit_vertex_parameter : require

layout (location = 0) in flat uint iIDFlat;
layout (location = 1) in __explicitInterpAMD uint iID;
layout (location = 2) in vec2 iUV;
layout (location = 3) in flat uint iMeshID;

layout(binding = 2) uniform sampler2D screen_tex;

layout (location = 0) out uvec2 result;
layout (location = 1) out uvec4 oUV;

void main() { 
	uint id0 = interpolateAtVertexAMD(iID, 0);
	uint id1 = interpolateAtVertexAMD(iID, 1);
	uint id2 = interpolateAtVertexAMD(iID, 2);

	vec3 barycentrics;
	if (iIDFlat == id0)
	{
		barycentrics.y = gl_BaryCoordSmoothAMD.x;
		barycentrics.z = gl_BaryCoordSmoothAMD.y;
		barycentrics.x = 1.0 - barycentrics.z - barycentrics.y;
	}
	else if (iIDFlat == id1)
	{
		barycentrics.x = gl_BaryCoordSmoothAMD.x;
		barycentrics.y = gl_BaryCoordSmoothAMD.y;
		barycentrics.z = 1.0 - barycentrics.x - barycentrics.y;
	}
	else if (iIDFlat == id2)
	{
		barycentrics.z = gl_BaryCoordSmoothAMD.x;
		barycentrics.x = gl_BaryCoordSmoothAMD.y;
		barycentrics.y = 1.0 - barycentrics.x - barycentrics.z;
	}
	else
	{
		barycentrics = vec3(1.0);
	}
	
	vec2 lod_result = textureQueryLod(screen_tex, iUV);
	uint packed_bary = packUnorm2x16(barycentrics.xy);
	uint tri_ID = iIDFlat;
	float clamped_lod = clamp(lod_result.x, 0, 9);
	uint lod_tri_ID = bitfieldInsert(tri_ID, uint(clamped_lod), 28, 4); 
	result = uvec2(iMeshID, lod_tri_ID);
	precise uvec2 uv = uvec2(floor(iUV * textureSize(screen_tex, int(clamped_lod))));
	oUV = uvec4(uv.x, uv.y, packed_bary, int(clamped_lod));
}