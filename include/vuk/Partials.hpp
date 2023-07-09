#pragma once

#include "vuk/AllocatorHelpers.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Future.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/SourceLocation.hpp"
#include <math.h>
#include <span>

namespace vuk {
	/// @brief Fill a buffer with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param buffer Buffer to fill
	/// @param src_data pointer to source data
	/// @param size size of source data
	inline Future host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, const void* src_data, size_t size) {
		// host-mapped buffers just get memcpys
		if (dst.mapped_ptr) {
			memcpy(dst.mapped_ptr, src_data, size);
			return { std::move(dst) };
		}

		auto src = *allocate_buffer(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, 1 });
		::memcpy(src->mapped_ptr, src_data, size);

		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("host_data_to_buffer");
		rgp->add_pass({ .name = "BUFFER UPLOAD",
		                .execute_on = copy_domain,
		                .resources = { "_dst"_buffer >> vuk::Access::eTransferWrite, "_src"_buffer >> vuk::Access::eTransferRead },
		                .execute = [size](vuk::CommandBuffer& command_buffer) {
			                command_buffer.copy_buffer("_src", "_dst", size);
		                } });
		rgp->attach_buffer("_src", *src, vuk::Access::eNone);
		rgp->attach_buffer("_dst", dst, vuk::Access::eNone);
		return { std::move(rgp), "_dst+" };
	}

	/// @brief Fill a buffer with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param dst Buffer to fill
	/// @param data source data
	template<class T>
	Future host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, std::span<T> data) {
		return host_data_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes());
	}

	/// @brief Download a buffer to GPUtoCPU memory
	/// @param buffer_src Buffer to download
	inline Future download_buffer(Future buffer_src) {
		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("download_buffer");
		rgp->attach_in("src", std::move(buffer_src));
		rgp->attach_buffer("dst", Buffer{ .memory_usage = MemoryUsage::eGPUtoCPU });
		rgp->inference_rule("dst", same_size_as("src"));
		rgp->add_pass(
		    { .name = "copy", .resources = { "src"_buffer >> eTransferRead, "dst"_buffer >> eTransferWrite }, .execute = [](vuk::CommandBuffer& command_buffer) {
			     command_buffer.copy_buffer("src", "dst", VK_WHOLE_SIZE);
		     } });
		return { rgp, "dst+" };
	}

	/// @brief Fill an image with host data
	/// @param allocator Allocator to use for temporary allocations
	/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
	/// @param image ImageAttachment to fill
	/// @param src_data pointer to source data
	inline Future host_data_to_image(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment image, const void* src_data) {
		size_t alignment = format_to_texel_block_size(image.format);
		assert(image.extent.sizing == Sizing::eAbsolute);
		size_t size = compute_image_size(image.format, static_cast<Extent3D>(image.extent.extent));
		auto src = *allocate_buffer(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		::memcpy(src->mapped_ptr, src_data, size);

		BufferImageCopy bc;
		bc.imageOffset = { 0, 0, 0 };
		bc.bufferRowLength = 0;
		bc.bufferImageHeight = 0;
		bc.imageExtent = static_cast<Extent3D>(image.extent.extent);
		bc.imageSubresource.aspectMask = format_to_aspect(image.format);
		bc.imageSubresource.mipLevel = image.base_level;
		bc.imageSubresource.baseArrayLayer = image.base_layer;
		assert(image.layer_count == 1); // unsupported yet
		bc.imageSubresource.layerCount = image.layer_count;

		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("host_data_to_image");
		rgp->add_pass({ .name = "IMAGE UPLOAD",
		                .execute_on = copy_domain,
		                .resources = { "_dst"_image >> vuk::Access::eTransferWrite, "_src"_buffer >> vuk::Access::eTransferRead },
		                .execute = [bc](vuk::CommandBuffer& command_buffer) {
			                command_buffer.copy_buffer_to_image("_src", "_dst", bc);
		                } });
		rgp->attach_buffer("_src", *src, vuk::Access::eNone);
		rgp->attach_image("_dst", image, vuk::Access::eNone);
		return { std::move(rgp), "_dst+" };
	}

	/// @brief Transition image for given access - useful to force certain access across different RenderGraphs linked by Futures
	/// @param image input Future of ImageAttachment
	/// @param dst_access Access to have in the future
	inline Future transition(Future image, Access dst_access) {
		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("transition");
		rgp->add_pass({ .name = "TRANSITION",
		                .execute_on = DomainFlagBits::eDevice,
		                .resources = { "_src"_image >> dst_access >> "_src+" },
		                .type = PassType::eForcedAccess });
		rgp->attach_in("_src", std::move(image));
		return { std::move(rgp), "_src+" };
	}

	/// @brief Generate mips for given ImageAttachment
	/// @param image input Future of ImageAttachment
	/// @param base_mip source mip level
	/// @param num_mips number of mip levels to generate
	inline Future generate_mips(Future image, uint32_t base_mip, uint32_t num_mips) {
		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("generate_mips");
		rgp->attach_in("_src", std::move(image));
		Name mip = Name("_mip_");

		std::vector<Name> diverged_names;
		for (uint32_t miplevel = base_mip; miplevel < (base_mip + num_mips); miplevel++) {
			Name div_name = mip.append(std::to_string(miplevel));
			if (miplevel != base_mip) {
				diverged_names.push_back(div_name.append("+"));
			} else {
				diverged_names.push_back(div_name);
			}
			rgp->diverge_image("_src", { .base_level = miplevel, .level_count = 1 }, div_name);
		}

		for (uint32_t miplevel = base_mip + 1; miplevel < (base_mip + num_mips); miplevel++) {
			uint32_t dmiplevel = miplevel - base_mip;

			Name mip_src_name = mip.append(std::to_string(miplevel - 1));
			auto mip_dst = std::to_string(miplevel);
			Name mip_dst_name = mip.append(std::to_string(miplevel));
			if (miplevel != base_mip + 1) {
				mip_src_name = mip_src_name.append("+");
			}
			Resource src_res(mip_src_name, Resource::Type::eImage, Access::eTransferRead);
			Resource dst_res(mip_dst_name, Resource::Type::eImage, Access::eTransferWrite, mip_dst_name.append("+"));
			rgp->add_pass({ .name = Name("MIP").append(mip_dst),
			                .execute_on = DomainFlagBits::eGraphicsOnGraphics,
			                .resources = { src_res, dst_res },
			                .execute = [mip_src_name, mip_dst_name, dmiplevel, miplevel](CommandBuffer& command_buffer) {
				                ImageBlit blit;
				                auto src_ia = *command_buffer.get_resource_image_attachment(mip_src_name);
				                auto dim = src_ia.extent;
				                assert(dim.sizing == Sizing::eAbsolute);
				                auto extent = dim.extent;
				                blit.srcSubresource.aspectMask = format_to_aspect(src_ia.format);
				                blit.srcSubresource.baseArrayLayer = src_ia.base_layer;
				                blit.srcSubresource.layerCount = src_ia.layer_count;
				                blit.srcSubresource.mipLevel = miplevel - 1;
				                blit.srcOffsets[0] = Offset3D{ 0 };
				                blit.srcOffsets[1] = Offset3D{ std::max((int32_t)extent.width >> (dmiplevel - 1), 1),
					                                             std::max((int32_t)extent.height >> (dmiplevel - 1), 1),
					                                             (int32_t)1 };
				                blit.dstSubresource = blit.srcSubresource;
				                blit.dstSubresource.mipLevel = miplevel;
				                blit.dstOffsets[0] = Offset3D{ 0 };
				                blit.dstOffsets[1] =
				                    Offset3D{ std::max((int32_t)extent.width >> (dmiplevel), 1), std::max((int32_t)extent.height >> (dmiplevel), 1), (int32_t)1 };
				                command_buffer.blit_image(mip_src_name, mip_dst_name, blit, Filter::eLinear);
			                } });
		}

		rgp->converge_image_explicit(diverged_names, "_src+");
		return { std::move(rgp), "_src+" };
	}

	/// @brief Allocates & fills a buffer with explicitly managed lifetime
	/// @param allocator Allocator to allocate this Buffer from
	/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
	template<class T>
	std::pair<Unique<Buffer>, Future> create_buffer(Allocator& allocator, vuk::MemoryUsage memory_usage, DomainFlagBits domain, std::span<T> data) {
		Unique<Buffer> buf(allocator);
		BufferCreateInfo bci{ memory_usage, sizeof(T) * data.size(), 1 };
		auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
		Buffer b = buf.get();
		return { std::move(buf), host_data_to_buffer(allocator, domain, b, data) };
	}

	/// @brief Allocates & fills an image, creates default ImageView for it (legacy)
	/// @param allocator Allocator to allocate this Texture from
	/// @param format Format of the image
	/// @param extent Extent3D of the image
	/// @param data pointer to data to fill the image with
	/// @param should_generate_mips if true, all mip levels are generated from the 0th level
	inline std::pair<Texture, Future> create_texture(Allocator& allocator, Format format, Extent3D extent, void* data, bool should_generate_mips, SourceLocationAtFrame loc = VUK_HERE_AND_NOW()) {
		ImageCreateInfo ici;
		ici.format = format;
		ici.extent = extent;
		ici.samples = Samples::e1;
		ici.initialLayout = ImageLayout::eUndefined;
		ici.tiling = ImageTiling::eOptimal;
		ici.usage = ImageUsageFlagBits::eTransferSrc | ImageUsageFlagBits::eTransferDst | ImageUsageFlagBits::eSampled;
		ici.mipLevels = should_generate_mips ? (uint32_t)log2f((float)std::max(extent.width, extent.height)) + 1 : 1;
		ici.arrayLayers = 1;
		auto tex = allocator.get_context().allocate_texture(allocator, ici, loc);

		auto upload_fut = host_data_to_image(allocator, DomainFlagBits::eTransferQueue, ImageAttachment::from_texture(tex), data);
		auto mipgen_fut = should_generate_mips ? generate_mips(std::move(upload_fut), 0, ici.mipLevels) : std::move(upload_fut);
		std::shared_ptr<RenderGraph> rgp = std::make_shared<RenderGraph>("create_texture");
		rgp->add_pass({ .name = "TRANSITION",
		                .execute_on = DomainFlagBits::eGraphicsQueue,
		                .resources = { "_src"_image >> Access::eFragmentSampled >> "_src+" },
		                .type = PassType::eForcedAccess });
		rgp->attach_in("_src", std::move(mipgen_fut));
		auto on_gfx = Future{ std::move(rgp), "_src+" };

		return { std::move(tex), std::move(on_gfx) };
	}
} // namespace vuk
