#pragma once

#include <vuk/vuk_fwd.hpp>

namespace vuk {
	/// @brief Simplest host memory allocator, uses new[] and delete[] to acquire enough memory
	struct NewDeleteAllocator {
		std::byte* allocate(size_t size) {
			return new std::byte[size];
		}
		void deallocate(std::byte* ptr) {
			delete[] ptr;
		}
	};

	struct CommandBufferCreateInfo {
		VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	};

	struct SemaphoreCreateInfo {
		bool timeline;
		uint64_t initial_value = 0;
	};

	/// @brief Thread-safe, global allocator
	struct GlobalAllocator {
		GlobalAllocator(Context& ctx);

		// direct
		VkFence allocate_fence(uint64_t absolute_frame, SourceLocation);
		VkCommandBuffer allocate_command_buffer(VkCommandBufferLevel, uint64_t absolute_frame, SourceLocation);
		VkSemaphore allocate_semaphore(uint64_t absolute_frame, SourceLocation);
		Buffer allocate_buffer(const BufferAllocationCreateInfo&, uint64_t absolute_frame, SourceLocation);
		Image allocate_image(const ImageCreateInfo&, uint64_t absolute_frame, SourceLocation);
		ImageView allocate_image_view(const ImageViewCreateInfo&, uint64_t absolute_frame, SourceLocation);

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

		void deallocate(VkPipeline);
		void deallocate(vuk::Image);
		void deallocate(vuk::ImageView);
		void deallocate(vuk::Buffer);
		void deallocate(vuk::PersistentDescriptorSet);

		VkFence create(const create_info_t<VkFence>& cinfo);
		VkSemaphore create(const create_info_t<VkSemaphore>& cinfo);
		ShaderModule create(const create_info_t<ShaderModule>& cinfo);
		VkCommandBuffer create(const create_info_t<VkCommandBuffer>& cinfo);
		PipelineBaseInfo create(const create_info_t<PipelineBaseInfo>& cinfo);
		VkPipelineLayout create(const create_info_t<VkPipelineLayout>& cinfo);
		DescriptorSetLayoutAllocInfo create(const create_info_t<DescriptorSetLayoutAllocInfo>& cinfo);
		ComputePipelineInfo create(const create_info_t<ComputePipelineInfo>& cinfo);
		PipelineInfo create(const create_info_t<PipelineInfo>& cinfo);
		VkRenderPass create(const create_info_t<VkRenderPass>& cinfo);
		VkFramebuffer create(const create_info_t<VkFramebuffer>& cinfo);
		Sampler create(const create_info_t<Sampler>& cinfo);
		DescriptorPool create(const create_info_t<DescriptorPool>& cinfo);
		DescriptorSet create(const create_info_t<vuk::DescriptorSet>& cinfo, uint64_t absolute_frame);
		RGImage create(const create_info_t<vuk::RGImage>& cinfo);

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
		void destroy(DescriptorSet ds);
		void destroy(const struct PoolAllocator& v);
		void destroy(const struct LinearAllocator& v);

		/// @brief Create a wrapped handle type (eg. a vuk::ImageView) from an externally sourced Vulkan handle
		/// @tparam T Vulkan handle type to wrap
		/// @param payload Vulkan handle to wrap
		/// @return The wrapped handle.
		template<class T>
		Handle<T> wrap(T payload);

		VkDevice device;
		struct DeviceMemoryAllocator* device_memory_allocator = nullptr;
		NewDeleteAllocator* host_memory_allocator = nullptr;
		DebugUtils* debug_utils = nullptr;

		std::atomic<uint64_t> unique_handle_id_counter = 0;

		struct GlobalAllocatorImpl* impl;

		VkPipelineCache vk_pipeline_cache = VK_NULL_HANDLE;
	};
}