#pragma once

#include <array>
#include <memory>
#include <vector>

#include "vuk/EmbeddedResource.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/Value.hpp"

namespace vuk::extra {
	VUK_EMBEDDED_RESOURCE(spd_cs_hlsl_spv);

	enum class ReductionType : uint32_t {
		Avg = 0,
		Min = 1,
		Max = 2,
	};

	/// @brief Generate all mips of an image using the Single Pass Downsampler
	/// @param image input Future of ImageAttachment
	/// @param type downsampling operator
	inline Value<ImageAttachment> generate_mips_spd(Value<ImageAttachment> image, ReductionType type = ReductionType::Avg) {
		PipelineBaseCreateInfo spd_pci;
		auto res = spd_cs_hlsl_spv();
		spd_pci.add_static_spirv((uint32_t*)res.data, res.size / 4, "spd.cs.hlsl");

		auto pass = vuk::make_pass("SPD", [type](CommandBuffer& command_buffer, VUK_IA(eComputeRW | eComputeSampled) src, VUK_ARG(PipelineBaseInfo*, eNone) pipeline) {
			// Collect details about the image
			auto extent = src->extent;
			auto mips = src->level_count;
			assert(mips <= 13);
			std::array<ImageAttachment, 13> mip_ia{};
			for (uint32_t i = 0; i < mips; ++i) {
				auto ia = static_cast<ImageAttachment>(src);
				ia.base_level = i;
				ia.level_count = 1;
				mip_ia[i] = ia;
			}
			Extent2D dispatch;
			dispatch.width = (extent.width + 63) >> 6;
			dispatch.height = (extent.height + 63) >> 6;

			// Bind source mip
			command_buffer.image_barrier(src, eComputeRW, eComputeSampled, 0, 1); // Prepare initial mip for read
			command_buffer.bind_compute_pipeline(pipeline);
			command_buffer.bind_image(0, 0, mip_ia[0]);
			switch (type) {
			case ReductionType::Avg:
				command_buffer.bind_sampler(0,
				                            0,
				                            {
				                                .minFilter = vuk::Filter::eLinear,
				                                .addressModeU = vuk::SamplerAddressMode::eClampToEdge,
				                                .addressModeV = vuk::SamplerAddressMode::eClampToEdge,
				                            });
				break;
			case ReductionType::Min:
				static const auto MinClampRMCI = VkSamplerReductionModeCreateInfo{
					.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
					.reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN,
				};
				command_buffer.bind_sampler(0,
				                            0,
				                            {
				                                .pNext = &MinClampRMCI,
				                                .minFilter = vuk::Filter::eLinear,
				                                .addressModeU = vuk::SamplerAddressMode::eClampToEdge,
				                                .addressModeV = vuk::SamplerAddressMode::eClampToEdge,
				                            });
				break;
			case ReductionType::Max:
				static const auto MaxClampRMCI = VkSamplerReductionModeCreateInfo{
					.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO,
					.reductionMode = VK_SAMPLER_REDUCTION_MODE_MAX,
				};
				command_buffer.bind_sampler(0,
				                            0,
				                            {
				                                .pNext = &MaxClampRMCI,
				                                .minFilter = vuk::Filter::eLinear,
				                                .addressModeU = vuk::SamplerAddressMode::eClampToEdge,
				                                .addressModeV = vuk::SamplerAddressMode::eClampToEdge,
				                            });
				break;
			}
			*command_buffer.scratch_buffer<uint32_t>(0, 1) = 0;
			// Bind target mips
			for (uint32_t i = 1; i < 13; i++)
				command_buffer.bind_image(0, i + 1, mip_ia[std::min(i, mips - 1)]);

			// Pass required shader data
			command_buffer.specialize_constants(0, mips - 1);
			command_buffer.specialize_constants(1, dispatch.width * dispatch.height);
			command_buffer.specialize_constants(2, extent.width);
			command_buffer.specialize_constants(3, extent.height);
			command_buffer.specialize_constants(4,
			                                    extent.width == extent.height && (extent.width & (extent.width - 1)) == 0 // Clever bitwise power-of-two check
			                                        ? 1u
			                                        : 0u);
			command_buffer.specialize_constants(5, static_cast<uint32_t>(type));
			command_buffer.specialize_constants(6, is_format_srgb(src->format) ? 1u : 0u);

			command_buffer.dispatch(dispatch.width, dispatch.height);
			command_buffer.image_barrier(src, eComputeSampled, eComputeRW, 0, 1); // Reconverge the image

			return src;
		});
		return pass(std::move(image), compile_pipeline(std::move(spd_pci)));
	}
} // namespace vuk::extra