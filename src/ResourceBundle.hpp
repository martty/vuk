#pragma once

#include <vuk/Context.hpp>
#include <vuk/Allocator.hpp>

namespace vuk {
	template<class Parent = Allocator>
	struct LinearResourceAllocator : Allocator {
		LinearResourceAllocator(Parent& allocator) : Allocator(allocator.ctx), parent(allocator) {}
		Parent& parent;
		uint32_t queue_family_index;
		VkCommandPool cpool = VK_NULL_HANDLE;
		std::vector<VkCommandBuffer> command_buffers;
		vuk::Buffer buffer;
		std::vector<vuk::Image> images;
		std::vector<vuk::ImageView> image_views;
		VkFence fence = VK_NULL_HANDLE;
		VkSemaphore sema = VK_NULL_HANDLE;
		LinearResourceAllocator* next = nullptr;

		LinearResourceAllocator clone();

		VkCommandBuffer allocate_command_buffer(VkCommandBufferLevel level, uint32_t queue_family_index) {
			if (cpool == VK_NULL_HANDLE) {
				VkCommandPoolCreateInfo cpci{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
				cpci.queueFamilyIndex = queue_family_index;
				this->queue_family_index = queue_family_index;
				cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
				assert(vkCreateCommandPool(ctx.device, &cpci, nullptr, &cpool) == VK_SUCCESS);
			}
			assert(queue_family_index == this->queue_family_index);
			VkCommandBuffer cbuf;
			VkCommandBufferAllocateInfo cbai{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
			cbai.commandPool = cpool;
			cbai.commandBufferCount = 1;
			cbai.level = level;

			assert(vkAllocateCommandBuffers(ctx.device, &cbai, &cbuf) == VK_SUCCESS);
			command_buffers.push_back(cbuf);
			return cbuf;
		}

		VkFence allocate_fence();
		VkSemaphore allocate_semaphore();
		VkSemaphore allocate_timeline_semaphore(uint64_t initial_value, uint64_t absolute_frame, SourceLocation) override;
		VkFramebuffer allocate_framebuffer(const struct FramebufferCreateInfo&);
		VkRenderPass allocate_renderpass(const struct RenderPassCreateInfo&);
		RGImage allocate_rendertarget(const struct RGCI&);
		Sampler allocate_sampler(const SamplerCreateInfo&);
		DescriptorSet allocate_descriptorset(const SetBinding&);
		PipelineInfo allocate_pipeline(const PipelineInstanceCreateInfo&);
		TimestampQuery register_timestamp_query(Query);
		vuk::ImageView allocate_image_view(const struct ImageViewCreateInfo&);

		TokenData& get_token_data(Token) override;
		void destroy(Token) override;

		Buffer allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment);
	};

	template<class Parent>
	LinearResourceAllocator<Parent> LinearResourceAllocator<Parent>::clone() {
		assert(0);
		return *ctx.impl->get_linear_allocator(queue_family_index);
		/*next = alloc;
		return alloc;*/
	}

	template<class Parent>
	VkFramebuffer LinearResourceAllocator<Parent>::allocate_framebuffer(const FramebufferCreateInfo& fbci) {
		//	return ctx->impl->framebuffer_cache.acquire(fbci);
		return{};
	}

	template<class Parent>
	VkSemaphore LinearResourceAllocator<Parent>::allocate_timeline_semaphore(uint64_t initial_value, uint64_t absolute_frame, SourceLocation) {
		VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
		VkSemaphoreTypeCreateInfo stci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
		stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
		sci.pNext = &stci;
		VkSemaphore sema;
		vkCreateSemaphore(ctx.device, &sci, nullptr, &sema);
		return sema;
	}

	template<class Parent>
	RGImage LinearResourceAllocator<Parent>::allocate_rendertarget(const RGCI& rgci) {
		//return ctx->impl->transient_images.acquire(rgci);
		return{};
	}

	template<class Parent>
	Sampler LinearResourceAllocator<Parent>::allocate_sampler(const SamplerCreateInfo& sci) {
		//return ctx->impl->sampler_cache.acquire(sci);
		return {};
	}

	template<class Parent>
	VkFence LinearResourceAllocator<Parent>::allocate_fence() {
		return ctx.impl->get_unpooled_fence();
	}
	
	template<class Parent>
	TimestampQuery LinearResourceAllocator<Parent>::register_timestamp_query(Query handle) {
		assert(0);
		return { 0 };
	}

	template<class Parent>
	PipelineInfo LinearResourceAllocator<Parent>::allocate_pipeline(const PipelineInstanceCreateInfo& pici) {
		//return ctx->impl->pipeline_cache.acquire(pici);
		return {};
	}
	
	template<class Parent>
	DescriptorSet LinearResourceAllocator<Parent>::allocate_descriptorset(const SetBinding&) {
		assert(0);
		return {};
	}
	
	template<class Parent>
	Buffer LinearResourceAllocator<Parent>::allocate_buffer(MemoryUsage mem_usage, BufferUsageFlags buffer_usage, size_t size, size_t alignment) {
		assert(0);
		return {};
	}

	template<class Parent>
	TokenData& LinearResourceAllocator<Parent>::get_token_data(Token t) {
		return parent.get_token_data(t);
	}

	template<class Parent>
	void LinearResourceAllocator<Parent>::destroy(Token t) {
		parent.destroy(t);
	}

	template<class Parent>
	ImageView LinearResourceAllocator<Parent>::allocate_image_view(const ImageViewCreateInfo& ivci) {
		assert(0);
		return {};
	}
}