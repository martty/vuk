#pragma once

#include "vuk/Context.hpp"
#include "vuk/RenderGraph.hpp"
#include <span>

namespace vuk {
	inline Future<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer buffer, void* src_data, size_t size) {
		// host-mapped buffers just get memcpys
		if (buffer.mapped_ptr) {
			memcpy(buffer.mapped_ptr, src_data, size);
			return { allocator, std::move(buffer) };
		}

		auto src = *allocate_buffer_cross_device(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, 1 });
		::memcpy(src->mapped_ptr, src_data, size);

		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		rgp->add_pass({ .name = "BUFFER UPLOAD",
		                .execute_on = copy_domain,
		                .resources = { "_dst"_buffer >> vuk::Access::eTransferWrite, "_src"_buffer >> vuk::Access::eTransferRead },
		                .execute = [size](vuk::CommandBuffer& command_buffer) {
			                command_buffer.copy_buffer("_src", "_dst", size);
		                } });
		rgp->attach_buffer("_src", *src, vuk::Access::eNone, vuk::Access::eNone);
		rgp->attach_buffer("_dst", buffer, vuk::Access::eNone, vuk::Access::eNone);
		return { allocator, std::move(rgp), "_dst+" };
	}

	template<class T>
	Future<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, std::span<T> data) {
		return host_data_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes());
	}

	inline Future<ImageAttachment> host_data_to_image(Allocator& allocator, DomainFlagBits copy_domain, ImageAttachment image, void* src_data) {
		size_t alignment = format_to_texel_block_size(image.format);
		size_t size = compute_image_size(image.format, static_cast<Extent3D>(image.extent));
		auto src = *allocate_buffer_cross_device(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		::memcpy(src->mapped_ptr, src_data, size);

		BufferImageCopy bc;
		bc.imageOffset = { 0, 0, 0 };
		bc.bufferRowLength = 0;
		bc.bufferImageHeight = 0;
		bc.imageExtent = static_cast<Extent3D>(image.extent);
		bc.imageSubresource.aspectMask = format_to_aspect(image.format);
		bc.imageSubresource.mipLevel = image.base_level;
		bc.imageSubresource.baseArrayLayer = image.base_layer;
		assert(image.layer_count == 1); // unsupported yet
		bc.imageSubresource.layerCount = image.layer_count;

		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		rgp->add_pass({ .name = "IMAGE UPLOAD",
		                .execute_on = copy_domain,
		                .resources = { "_dst"_image >> vuk::Access::eTransferWrite, "_src"_buffer >> vuk::Access::eTransferRead },
		                .execute = [bc](vuk::CommandBuffer& command_buffer) {
			                command_buffer.copy_buffer_to_image("_src", "_dst", bc);
		                } });
		rgp->attach_buffer("_src", *src, vuk::Access::eNone, vuk::Access::eNone);
		rgp->attach_image("_dst", image, vuk::Access::eNone, vuk::Access::eNone);
		return { allocator, std::move(rgp), "_dst+" };
	}

	inline Future<ImageAttachment> transition(Future<ImageAttachment> image, Access dst_access) {
		auto& allocator = image.get_allocator();
		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		rgp->add_pass({
			.name = "TRANSITION",
			.execute_on = DomainFlagBits::eDevice,
			.resources = {"_src"_image >> dst_access >> "_src+"} });
		rgp->attach_in("_src", std::move(image), Access::eNone);
		return { allocator, std::move(rgp), "_src+" };
	}

	/*inline Future<ImageAttachment> acquire_to_queue(Future<ImageAttachment> image, DomainFlagBits dst_domain) {
		auto& allocator = image.get_allocator();
		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		rgp->attach_in("_src", std::move(image), Access::eNone);
		return { allocator, std::move(rgp), "_src+" };
	}*/

	/// @brief Allocates & fills a buffer with explicitly managed lifetime
	/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
	/// @param buffer_usage How this buffer will be used (since data is provided, TransferDst is added to the flags)
	/// @return The allocated Buffer
	template<class T>
	std::pair<Unique<BufferCrossDevice>, Future<Buffer>> create_buffer_cross_device(Allocator& allocator, MemoryUsage mem_usage, std::span<T> data) {
		Unique<BufferCrossDevice> buf(allocator);
		BufferCreateInfo bci{ mem_usage, sizeof(T) * data.size(), 1 };
		auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
		memcpy(buf->mapped_ptr, data.data(), data.size_bytes());
		Buffer b = buf.get();
		return { std::move(buf), Future<Buffer>{ allocator, std::move(b) } };
	}

	template<class T>
	std::pair<Unique<BufferGPU>, Future<Buffer>> create_buffer_gpu(Allocator& allocator, DomainFlagBits domain, std::span<T> data) {
		Unique<BufferGPU> buf(allocator);
		BufferCreateInfo bci{ MemoryUsage::eGPUonly, sizeof(T) * data.size(), 1 };
		auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
		Buffer b = buf.get();
		return { std::move(buf), host_data_to_buffer(allocator, domain, b, data) };
	}

	inline std::pair<Texture, Future<ImageAttachment>> create_texture(Allocator& allocator, Format format, Extent3D extent, void* data, bool generate_mips) {
		ImageCreateInfo ici;
		ici.format = format;
		ici.extent = extent;
		ici.samples = Samples::e1;
		ici.initialLayout = ImageLayout::eUndefined;
		ici.tiling = ImageTiling::eOptimal;
		ici.usage = ImageUsageFlagBits::eTransferRead | ImageUsageFlagBits::eTransferWrite | ImageUsageFlagBits::eSampled;
		ici.mipLevels = generate_mips ? (uint32_t)log2f((float)std::max(extent.width, extent.height)) + 1 : 1;
		ici.arrayLayers = 1;
		auto tex = allocator.get_context().allocate_texture(allocator, ici);

		auto upload_fut = host_data_to_image(allocator, DomainFlagBits::eTransferQueue, ImageAttachment::from_texture(tex), data);
		std::unique_ptr<RenderGraph> rgp = std::make_unique<RenderGraph>();
		rgp->add_pass({
			.name = "TRANSITION",
			.execute_on = DomainFlagBits::eGraphicsQueue,
			.resources = {"_src"_image >> Access::eFragmentSampled >> "_src+"} });
		rgp->attach_in("_src", std::move(upload_fut), Access::eNone);
		auto on_gfx = Future<ImageAttachment>{ allocator, std::move(rgp), "_src+" };
		
		return { std::move(tex), std::move(on_gfx) };
	}
} // namespace vuk
