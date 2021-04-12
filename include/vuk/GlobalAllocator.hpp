#pragma once

#include <vuk/vuk_fwd.hpp>
#include <vuk/Config.hpp>
#include <vuk/Image.hpp>
#include <vuk/Buffer.hpp>
#include <vuk/Pipeline.hpp>
#include <vuk/SampledImage.hpp>
#include <vuk/Allocator.hpp>
#include <Allocator.hpp>

namespace vuk {
	struct Token;

	/// @brief Simplest host memory allocator, uses new[] and delete[] to acquire enough memory
	struct NewDeleteAllocator {
		std::byte* allocate(size_t size) {
			return new std::byte[size];
		}
		void deallocate(std::byte* ptr) {
			delete[] ptr;
		}
	};

	struct FenceCreateInfo {
	};

	struct CommandBufferCreateInfo {
		VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		uint32_t queue_family_index;
	};

	struct SemaphoreCreateInfo {
		bool timeline;
		uint64_t initial_value = 0;
	};

	struct BufferAllocationCreateInfo {
		/// @brief mem_usage Determines which memory will be used.
		MemoryUsage mem_usage;
		/// @param buffer_usage Set to the usage of the buffer.
		BufferUsageFlags buffer_usage;
		/// @param size Size of the allocation.
		size_t size;
		/// @param alignment Minimum alignment of the allocation.
		size_t alignment;
	};

	/// @brief Thread-safe, global allocator
	struct GlobalAllocator final : public Allocator {
		GlobalAllocator(Context& ctx);

		// direct
		VkFence allocate_fence(uint64_t absolute_frame, SourceLocation);
		VkCommandBuffer allocate_command_buffer(const CommandBufferCreateInfo&, uint64_t absolute_frame, SourceLocation);
		VkSemaphore allocate_semaphore(uint64_t absolute_frame, SourceLocation);
		VkSemaphore allocate_timeline_semaphore(uint64_t initial_value, uint64_t absolute_frame, SourceLocation) override;
		Unique<Buffer> allocate_buffer(const BufferAllocationCreateInfo&, uint64_t absolute_frame, SourceLocation);
		Unique<Image> allocate_image(const struct ImageCreateInfo&, uint64_t absolute_frame, SourceLocation);
		Unique<ImageView> allocate_image_view(const struct ImageViewCreateInfo&, uint64_t absolute_frame, SourceLocation);
		Token allocate_token();

		/// @brief Allocate host memory (~malloc)
		/// @param  Struct describing the size
		/// @return Pointer to allocated memory
		std::byte* allocate_host(size_t size, uint64_t absolute_frame, SourceLocation);
		// cached
		VkFramebuffer allocate_framebuffer(const struct FramebufferCreateInfo&, uint64_t absolute_frame, SourceLocation);
		VkRenderPass allocate_renderpass(const struct RenderPassCreateInfo&, uint64_t absolute_frame, SourceLocation);
		RGImage allocate_rendertarget(const struct RGCI&, uint64_t absolute_frame, SourceLocation);
		Sampler allocate_sampler(const SamplerCreateInfo&, uint64_t absolute_frame, SourceLocation);
		DescriptorSet allocate_descriptorset(const SetBinding&, uint64_t absolute_frame, SourceLocation);
		PipelineInfo allocate_pipeline(const PipelineInstanceCreateInfo&, uint64_t absolute_frame, SourceLocation);
		PipelineBaseInfo& allocate_pipeline_base(const PipelineBaseCreateInfo&, uint64_t absolute_frame, SourceLocation);

		// legacy
		Texture allocate_texture(const vuk::ImageCreateInfo& ici, uint64_t absolute_frame, SourceLocation);
		std::pair<Texture, Token> create_texture(vuk::Format format, vuk::Extent3D extent, void* data, bool generate_mips, uint64_t absolute_frame, SourceLocation);

		void deallocate(VkPipeline);
		void deallocate(vuk::Image);
		void deallocate(vuk::ImageView);
		void deallocate(VkImageView);
		void deallocate(vuk::Buffer);
		void deallocate(vuk::PersistentDescriptorSet);

		VkFence create(const FenceCreateInfo& cinfo, uint64_t absolute_frame);
		VkSemaphore create(const SemaphoreCreateInfo& cinfo, uint64_t absolute_frame);
		ShaderModule create(const create_info_t<ShaderModule>& cinfo, uint64_t absolute_frame);
		VkCommandBuffer create(const CommandBufferCreateInfo& cinfo, uint64_t absolute_frame);
		PipelineBaseInfo create(const create_info_t<PipelineBaseInfo>& cinfo, uint64_t absolute_frame);
		VkPipelineLayout create(const create_info_t<VkPipelineLayout>& cinfo, uint64_t absolute_frame);
		DescriptorSetLayoutAllocInfo create(const create_info_t<DescriptorSetLayoutAllocInfo>& cinfo, uint64_t absolute_frame);
		ComputePipelineInfo create(const create_info_t<ComputePipelineInfo>& cinfo, uint64_t absolute_frame);
		PipelineInfo create(const create_info_t<PipelineInfo>& cinfo, uint64_t absolute_frame);
		VkRenderPass create(const create_info_t<VkRenderPass>& cinfo, uint64_t absolute_frame);
		VkFramebuffer create(const create_info_t<VkFramebuffer>& cinfo, uint64_t absolute_frame);
		Sampler create(const create_info_t<Sampler>& cinfo, uint64_t absolute_frame);
		DescriptorPool create(const create_info_t<DescriptorPool>& cinfo, uint64_t absolute_frame);
		DescriptorSet create(const create_info_t<vuk::DescriptorSet>& cinfo, uint64_t absolute_frame);
		RGImage create(const create_info_t<vuk::RGImage>& cinfo, uint64_t absolute_frame);
		ImageView create(const ImageViewCreateInfo& cinfo, uint64_t absolute_frame);
		struct LinearAllocator create(const create_info_t<struct LinearAllocator>& cinfo, uint64_t absolute_frame);

		void destroy(const struct RGImage& image);
		void destroy(const DescriptorPool& dp);
		void destroy(const PipelineInfo& pi);
		void destroy(const ShaderModule& sm);
		void destroy(const DescriptorSetLayoutAllocInfo& ds);
		void destroy(const VkPipelineLayout& pl);
		void destroy(const VkRenderPass& rp);
		void destroy(const DescriptorSet&);
		void destroy(const VkFramebuffer& fb);
		void destroy(const Sampler& sa);
		void destroy(const PipelineBaseInfo& pbi);
		void destroy(VkSemaphore);
		void destroy(VkFence);
		void destroy(Image image);
		void destroy(ImageView image);
		void destroy(const Buffer& buffer);
		void destroy(const struct PoolAllocator& v);
		void destroy(const struct LinearAllocator& v);

		struct TokenData& get_token_data(Token) override;
		void destroy(Token) override;

		/// @brief Create a wrapped handle type (eg. a vuk::ImageView) from an externally sourced Vulkan handle
		/// @tparam T Vulkan handle type to wrap
		/// @param payload Vulkan handle to wrap
		/// @return The wrapped handle.
		template<class T>
		Handle<T> wrap(T payload);

		VkDevice device;
		struct DeviceMemoryAllocator* device_memory_allocator = nullptr;
		NewDeleteAllocator* host_memory_allocator = nullptr;
		struct DebugUtils* debug_utils = nullptr;

		std::atomic<uint64_t> unique_handle_id_counter = 0;

		struct GlobalAllocatorImpl* impl;

		VkPipelineCache vk_pipeline_cache = VK_NULL_HANDLE;
		void load_pipeline_cache(std::span<std::byte> data);
		std::vector<std::byte> save_pipeline_cache();
	};

	template<class T>
	Handle<T> GlobalAllocator::wrap(T payload) {
		return { { unique_handle_id_counter++ }, payload };
	}
}