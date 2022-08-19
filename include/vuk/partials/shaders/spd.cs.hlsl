#define A_GPU 1
#define A_HLSL 1
#define A_HALF
#define SPD_PACKED_ONLY
#define SPD_LINEAR_SAMPLER
#include "ffx_a.h"

struct SpdGlobalAtomicBuffer {
	uint counter;
};

[[vk::binding(0)]][[vk::combinedImageSampler]] Texture2D<float4> imgSrc;
[[vk::binding(0)]][[vk::combinedImageSampler]] SamplerState srcSampler;
[[vk::binding(1)]] globallycoherent RWStructuredBuffer<SpdGlobalAtomicBuffer> spdGlobalAtomic;
[[vk::binding(2)]] RWTexture2D<float4> imgDst0;
[[vk::binding(3)]] RWTexture2D<float4> imgDst1;
[[vk::binding(4)]] RWTexture2D<float4> imgDst2;
[[vk::binding(5)]] RWTexture2D<float4> imgDst3;
[[vk::binding(6)]] RWTexture2D<float4> imgDst4;
[[vk::binding(7)]] globallycoherent RWTexture2D<float4> imgDst5;
[[vk::binding(8)]] RWTexture2D<float4> imgDst6;
[[vk::binding(9)]] RWTexture2D<float4> imgDst7;
[[vk::binding(10)]] RWTexture2D<float4> imgDst8;
[[vk::binding(11)]] RWTexture2D<float4> imgDst9;
[[vk::binding(12)]] RWTexture2D<float4> imgDst10;
[[vk::binding(13)]] RWTexture2D<float4> imgDst11;

[[vk::constant_id(0)]] const uint Mips = 0;
[[vk::constant_id(1)]] const uint NumWorkGroups = 0;

[[vk::constant_id(2)]] const uint InputWidth = 0;
[[vk::constant_id(3)]] const uint InputHeight = 0;
static const uint2 InputSize = {InputWidth, InputHeight};
[[vk::constant_id(4)]] const uint InputPOT = 0; // 1 = input is a power-of-two square

static const float InvInputWidth = 1.0 / float(InputWidth);
static const float InvInputHeight = 1.0 / float(InputHeight);
static const float2 InvInputSize = {InvInputWidth, InvInputHeight};

static const uint ReductionTypeAvg = 0;
static const uint ReductionTypeMin = 1;
static const uint ReductionTypeMax = 2;
[[vk::constant_id(5)]] const uint ReductionType = 0;
[[vk::constant_id(6)]] const uint SRGB = 0; // 1 = sRGB load/store enabled

groupshared AH2 spdIntermediateRG[16][16];
groupshared AH2 spdIntermediateBA[16][16];
groupshared AU1 spdCounter;

RWTexture2D<float4> GetDstMip(AU1 level) {
	switch(level) {
		case 0: return imgDst0;
		case 1: return imgDst1;
		case 2: return imgDst2;
		case 3: return imgDst3;
		case 4: return imgDst4;
		case 5: return imgDst5;
		case 6: return imgDst6;
		case 7: return imgDst7;
		case 8: return imgDst8;
		case 9: return imgDst9;
		case 10: return imgDst10;
		default /*11*/: return imgDst11;
	}
}

void SpdIncreaseAtomicCounter(AU1 slice) {
	InterlockedAdd(spdGlobalAtomic[0].counter, 1, spdCounter);
}
AU1 SpdGetAtomicCounter() {
	return spdCounter;
}
void SpdResetAtomicCounter(AU1 slice) {
	spdGlobalAtomic[0].counter = 0;
}
AH4 SpdLoadSourceImageH(ASU2 p, AU1 slice){
	AF2 textureCoord = p * InvInputSize + InvInputSize;
	return AH4(imgSrc.SampleLevel(srcSampler, textureCoord, 0));
}
AH4 SpdLoadH(ASU2 tex, AU1 slice) {
	if (!InputPOT)
		tex = min(tex, InputSize >> 6);
	AH4 value = AH4(imgDst5[tex]);
	if (SRGB)
		value *= value;
	return value;
}
void SpdStoreH(ASU2 pix, AH4 value, AU1 index, AU1 slice) {
	if (SRGB)
		value = sqrt(value);
	GetDstMip(index)[pix] = AF4(value);
}
AH4 SpdLoadIntermediateH(AU1 x, AU1 y) {
	return AH4(
		spdIntermediateRG[x][y],
		spdIntermediateBA[x][y]
	);
}
void SpdStoreIntermediateH(AU1 x, AU1 y, AH4 value) {
	spdIntermediateRG[x][y] = value.rg;
	spdIntermediateBA[x][y] = value.ba;
}
AH4 SpdReduce4H(AH4 v0, AH4 v1, AH4 v2, AH4 v3) {
	if (ReductionType == ReductionTypeAvg)
		return (v0+v1+v2+v3) * AH1(0.25);
	if (ReductionType == ReductionTypeMin)
		return min(min(v0,v1),min(v2,v3));
	if (ReductionType == ReductionTypeMax)
		return max(max(v0,v1),max(v2,v3));
	return AH4(0,0,0,0); // Something unimplemented
}

#if 0
// Single precision functions
AF4 SpdLoadSourceImage(ASU2 p, AU1 slice) {
	AF2 textureCoord = p * InvInputSize + InvInputSize;
	return imgSrc.SampleLevel(srcSampler, textureCoord, 0);
}
AF4 SpdLoad(ASU2 tex, AU1 slice) {
	if (!InputPOT)
		tex = min(tex, InputSize >> 6);
	AF4 value = imgDst5[tex];
	if (SRGB)
		value *= value;
	return value;
}
void SpdStore(ASU2 pix, AF4 value, AU1 mip, AU1 slice) {
	if (SRGB)
		value = sqrt(value);
	GetDstMip(mip)[pix] = value;
}
AF4 SpdLoadIntermediate(AU1 x, AU1 y) {
	return AF4(
		spdIntermediateR[x][y],
		spdIntermediateG[x][y],
		spdIntermediateB[x][y],
		spdIntermediateA[x][y]
	);
}
void SpdStoreIntermediate(AU1 x, AU1 y, AF4 value) {
	spdIntermediateR[x][y] = value.r;
	spdIntermediateG[x][y] = value.g;
	spdIntermediateB[x][y] = value.b;
	spdIntermediateA[x][y] = value.a;
}
AF4 SpdReduce4(AF4 v0, AF4 v1, AF4 v2, AF4 v3) {
	if (ReductionType == ReductionTypeAvg)
		return (v0+v1+v2+v3) * 0.25;
	if (ReductionType == ReductionTypeMin)
		return min(min(v0,v1),min(v2,v3));
	if (ReductionType == ReductionTypeMax)
		return max(max(v0,v1),max(v2,v3));
	return AH4(0,0,0,0); // Something unimplemented
}
#endif

#include "ffx_spd.h"

[numthreads(256, 1, 1)]
void main(uint3 WorkGroupId: SV_GroupID, uint LocalThreadIndex: SV_GroupIndex) {
	SpdDownsampleH(AU2(WorkGroupId.xy), AU1(LocalThreadIndex),  
		AU1(Mips), AU1(NumWorkGroups), AU1(WorkGroupId.z));
}
