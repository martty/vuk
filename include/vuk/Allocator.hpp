#pragma once

#include <vuk/vuk_fwd.hpp>
#include <vuk/Config.hpp>
#include <vuk/Result.hpp>
#include <vuk/Exception.hpp>
#include <span>
#include <vector>

namespace vuk {
	struct SourceLocation {
		const char* file;
		unsigned line;
	};
#define VUK_HERE() vuk::SourceLocation{__FILE__, __LINE__}

	struct AllocateException : vuk::Exception {
		AllocateException(VkResult res) {
			switch (res) {
			case VK_ERROR_OUT_OF_HOST_MEMORY:
			{
				error_message = "Out of host memory."; break;
			}
			default:
				assert(0 && "Unimplemented error."); break;
			}
		}
	};

	struct CPUResorce {
		virtual void* allocate_cpu(size_t bytes, size_t alignment, uint64_t frame, SourceLocation loc = {}) {
			return upstream->allocate_cpu(bytes, frame, alignment, loc);
		}
		virtual void deallocate_cpu(void* ptr, size_t bytes, size_t alignment) {
			return upstream->deallocate_cpu(ptr, bytes, alignment);
		}

		CPUResorce* upstream = nullptr;
	};

	struct VkResource {
		VkResource(class Context& ctx, VkResource* upstream) : ctx(ctx), upstream(upstream) {}

		/*virtual uint64_t allocate_gpu() {

		}*/

		//virtual VkCommandBuffer allocate_command_buffer(VkCommandBufferLevel, uint32_t queue_family_index, uint64_t frame, SourceLocation loc);
		virtual ImageView allocate_image_view(const ImageViewCreateInfo& info, uint64_t frame, SourceLocation loc) { return upstream->allocate_image_view(info, frame, loc); }
		virtual Sampler allocate_sampler(const SamplerCreateInfo& info, uint64_t frame, SourceLocation loc) { return upstream->allocate_sampler(info, frame, loc); }
		/*virtual DescriptorSet allocate_descriptorset(const SetBinding&, uint64_t frame, SourceLocation loc);
		virtual VkFramebuffer allocate_framebuffer(const struct FramebufferCreateInfo&, uint64_t frame, SourceLocation loc);
		virtual VkRenderPass allocate_renderpass(const struct RenderPassCreateInfo&, uint64_t frame, SourceLocation loc);*/

		virtual Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, uint64_t frame, SourceLocation loc) { return upstream->allocate_semaphores(dst, frame, loc); }
		virtual Result<void, AllocateException> allocate_timeline_semaphore(uint64_t initial_value, uint64_t frame, SourceLocation loc) { return upstream->allocate_timeline_semaphore(initial_value, frame, loc); }
		virtual Result<void, AllocateException> allocate_fence(uint64_t frame, SourceLocation loc) { return upstream->allocate_fence(frame, loc); }

		virtual void deallocate_image_view(ImageView iv) { upstream->deallocate_image_view(iv); }
		virtual void deallocate_sampler(Sampler samp) { upstream->deallocate_sampler(samp); }
		virtual void deallocate_semaphores(std::span<const VkSemaphore> sema) { upstream->deallocate_semaphores(sema); }
		virtual void deallocate_timeline_semaphore(VkSemaphore sema) { upstream->deallocate_timeline_semaphore(sema); }
		virtual void deallocate_fence(VkFence fence) { upstream->deallocate_fence(fence); }

		VkResource* upstream = nullptr;
		class Context& ctx;
	};

	struct Global : VkResource {
		Global(Context& ctx) : VkResource(ctx, nullptr) {}

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, uint64_t frame, SourceLocation loc) override {
			VkSemaphoreCreateInfo sci{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
			for (int64_t i = 0; i < (int64_t)dst.size(); i++) {
				VkResult res = vkCreateSemaphore(ctx.device, &sci, nullptr, &dst[i]);
				if (res != VK_SUCCESS) {
					// release resources that we already allocated to not leak
					for (i--; i >= 0; i--) {
						vkDestroySemaphore(ctx.device, dst[i], nullptr);
					}
					return { expected_error, AllocateException{res} };
				}
			}
			return { expected_value };
		}

		void deallocate_semaphores(std::span<const VkSemaphore> src) override {
			for (auto& v : src) {
				vkDestroySemaphore(ctx.device, v, nullptr);
			}
		}
	};

	struct NLinear : VkResource {
		using VkResource::VkResource;

		std::vector<VkSemaphore> semaphores;

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, uint64_t frame, SourceLocation loc) override {
			auto result = upstream->allocate_semaphores(dst, frame, loc);
			semaphores.insert(semaphores.end(), dst.begin(), dst.end());
			return result;
		}

		void deallocate_semaphores(std::span<const VkSemaphore>) override {} // linear allocator, noop

		~NLinear() {
			upstream->deallocate_semaphores(semaphores);
		}
	};

	struct NAllocator {
		explicit NAllocator(VkResource& mr) : mr(&mr) {}

		Result<void, AllocateException> allocate_semaphores(std::span<VkSemaphore> dst, uint64_t frame, SourceLocation loc) {
			return mr->allocate_semaphores(dst, frame, loc);
		}

		void deallocate_semaphores(std::span<const VkSemaphore> src) {
			mr->deallocate_semaphores(src);
		}

		VkResource* mr;
	};
}