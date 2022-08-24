#pragma once

#include "vuk/CommandBuffer.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/Future.hpp"
#include <vector>
#include <memory>
#include <array>

#include "shaders/spd.cs.spv.h"

namespace vuk {
	enum class ReductionType : uint32_t {
		Avg = 0,
		Min = 1,
		Max = 2,
	};

	/// @brief Generate all mips of an image using the Single Pass Downsampler
	/// @param image input Future of ImageAttachment
	/// @param type downsampling operator
	//TODO Eliminate need for Context param
	//TODO Generate the SPIR-V from HLSL sources at build-time
	inline Future generate_mips_spd(Context& ctx, Future image, ReductionType type = ReductionType::Avg) {
		static bool compiled = false;
		if (!compiled) {
			auto spd_pci = PipelineBaseCreateInfo();
			auto spd_ptr_start = reinterpret_cast<const uint32_t*>(spd_cs_spv);
			auto spd_ptr_end = spd_ptr_start + (spd_cs_spv_len / sizeof(uint32_t));
			auto spd_pci_vec = std::vector(spd_ptr_start, spd_ptr_end);
			spd_pci.add_spirv(std::move(spd_pci_vec), "spd.cs.hlsl");
			ctx.create_named_pipeline("VUK_SPD", spd_pci);
			compiled = true;
		}
		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("generate_mips_spd");
		rgp->attach_in("_src", std::move(image));
		rgp->add_pass({ .name = "SPD",
		                .resources = {
			                 "_src"_image >> eComputeRW, // transition target
			                 "_src"_image >> eComputeSampled, // additional usage
		                },
		                .execute = [type](CommandBuffer& command_buffer) {
			                // Collect details about the image
			                auto src_ia = *command_buffer.get_resource_image_attachment("_src");
			                auto extent = src_ia.extent.extent;
			                auto mips = src_ia.level_count;
			                assert(mips <= 13);
			                std::array<ImageAttachment, 13> mip_ia{};
			                for (uint32_t i = 0;i < mips; ++i) {
				                auto ia = src_ia;
				                ia.base_level = i;
				                ia.level_count = 1;
				                mip_ia[i] = ia;
			                }
			                Extent2D dispatch;
			                dispatch.width = (extent.width + 63) >> 6;
			                dispatch.height = (extent.height + 63) >> 6;

			                // Bind source mip
			                command_buffer.image_barrier("_src", eComputeRW, eComputeSampled, 0, 1); // Prepare initial mip for read
			                command_buffer.bind_compute_pipeline("VUK_SPD");
			                command_buffer.bind_image(0, 0, mip_ia[0], ImageLayout::eGeneral);
			                switch (type) {
			                case ReductionType::Avg:
				                command_buffer.bind_sampler(0, 0, {
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
				                command_buffer.bind_sampler(0, 0, {
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
				                command_buffer.bind_sampler(0, 0, {
					                .pNext = &MaxClampRMCI,
					                .minFilter = vuk::Filter::eLinear,
					                .addressModeU = vuk::SamplerAddressMode::eClampToEdge,
					                .addressModeV = vuk::SamplerAddressMode::eClampToEdge,
				                });
				                break;
			                }
			                *command_buffer.map_scratch_buffer<uint32_t>(0, 1) = 0;
			                // Bind target mips
			                for (uint32_t i = 1; i < 13; i++)
				                command_buffer.bind_image(0, i+1, mip_ia[std::min(i, mips-1)], ImageLayout::eGeneral);

			                // Pass required shader data
			                command_buffer.specialize_constants(0, mips-1);
			                command_buffer.specialize_constants(1, dispatch.width * dispatch.height);
			                command_buffer.specialize_constants(2, extent.width);
			                command_buffer.specialize_constants(3, extent.height);
			                command_buffer.specialize_constants(4,
				                extent.width == extent.height &&
				                (extent.width & (extent.width - 1)) == 0 // Clever bitwise power-of-two check
				                ? 1u : 0u);
			                command_buffer.specialize_constants(5, static_cast<uint32_t>(type));
			                command_buffer.specialize_constants(6, is_format_srgb(src_ia.format)? 1u : 0u);

			                command_buffer.dispatch(dispatch.width, dispatch.height);
			                command_buffer.image_barrier("_src", eComputeSampled, eComputeRW, 0, 1); // Reconverge the image
		                }
		});
		return { std::move(rgp), "_src+" };
	}
} // namespace vuk
