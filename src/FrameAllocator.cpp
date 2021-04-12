#include <vuk/FrameAllocator.hpp>
#include <vuk/GlobalAllocator.hpp>
#include <Allocator.hpp>
#include <ResourceBundle.hpp>
#include <array>
#include <vector>
#include <mutex>
#include <vuk/Types.hpp>
#include <vuk/Image.hpp>
#include <vuk/Buffer.hpp>
#include <Cache.hpp>
#include <vuk/Partials.hpp>

namespace vuk {
	struct FrameAllocatorImpl {
		FrameAllocatorImpl(RingFrameAllocator& rfa, uint64_t absolute_frame, unsigned frame) :
			fence_pools(rfa.fence_pools, frame),
			commandbuffer_pools(rfa.commandbuffer_pools, frame),
			tsquery_pools(rfa.tsquery_pools, frame),
			semaphore_pools(rfa.semaphore_pools, frame),
			sampled_images(rfa.sampled_images, frame),
			scratch_buffers(rfa.scratch_buffers, absolute_frame, frame),
			descriptor_sets(rfa.descriptor_sets, absolute_frame, frame),
			pool_cache(rfa.pool_cache)
		{}

		std::mutex recycle_lock;
		std::vector<vuk::Image> image_recycle;
		std::vector<VkImageView> image_view_recycle;
		std::vector<VkPipeline> pipeline_recycle;
		std::vector<vuk::Buffer> buffer_recycle;
		std::vector<vuk::PersistentDescriptorSet> pds_recycle;
		//std::vector<LinearResourceAllocator*> lra_recycle;

		PoolView<VkFence> fence_pools; // must be first, so we wait for the fences
		PoolView<VkCommandBuffer> commandbuffer_pools;
		PoolView<TimestampQuery> tsquery_pools;
		PoolView<VkSemaphore> semaphore_pools;
		PoolView<vuk::SampledImage> sampled_images;
		PerFrameCacheView<LinearAllocator> scratch_buffers;
		PerFrameCacheView<vuk::DescriptorSet> descriptor_sets;

		Cache<vuk::DescriptorPool>& pool_cache;
	};

	RingFrameAllocator::RingFrameAllocator(GlobalAllocator& ga, size_t FC) : ctx(ga.ctx),
		parent(ga), FC(FC),
		commandbuffer_pools(ga, FC),
		tsquery_pools(ga, FC),
		semaphore_pools(ga, FC),
		fence_pools(ga, FC),
		sampled_images(ga, FC),
		scratch_buffers(ga, FC),
		descriptor_sets(ga, FC),
		pool_cache(ga) {
		allocators = (FrameAllocatorImpl*) new char[sizeof(FrameAllocatorImpl) * FC];
		for (size_t i = 0; i < FC; i++) {
			new(&allocators[i]) FrameAllocatorImpl(*this, 0, i);
		}
	}

	// FrameAllocatorImpl must be known for dtor
	RingFrameAllocator::~RingFrameAllocator() = default;

	// --------------- FA --------------------

	FrameAllocator::FrameAllocator(RingFrameAllocator& rfa, uint64_t absolute_frame, unsigned frame)
		: Allocator(rfa.ctx), parent(rfa.parent), impl(&rfa.allocators[frame]), absolute_frame(absolute_frame), frame(frame), frames_in_flight(rfa.FC) {
		// extract query results before resetting
		/*std::unordered_map<uint64_t, uint64_t> query_results;
		for (auto& p : impl->tsquery_pools.per_frame_storage) {
			p.get_results(ctx);
			for (auto& [src, dst] : p.id_to_value_mapping) {
				query_results[src] = p.host_values[dst];
			}
		}

		impl->query_result_map = std::move(query_results);*/

		impl->fence_pools.reset(); // must be first, so we wait for the fences
		impl->commandbuffer_pools.reset();
		impl->tsquery_pools.reset();
		impl->semaphore_pools.reset();
		impl->sampled_images.reset();

		// image recycling
		for (auto& img : impl->image_recycle) {
			parent.deallocate(img);
		}
		impl->image_recycle.clear();

		for (auto& iv : impl->image_view_recycle) {
			parent.deallocate(iv);
		}
		impl->image_view_recycle.clear();

		for (auto& p : impl->pipeline_recycle) {
			parent.deallocate(p);
		}
		impl->pipeline_recycle.clear();

		for (auto& b : impl->buffer_recycle) {
			parent.deallocate(b);
		}
		impl->buffer_recycle.clear();

		for (auto& pds : impl->pds_recycle) {
			parent.deallocate(pds);
		}
		impl->pds_recycle.clear();

		for (auto& [k, v] : impl->scratch_buffers.cache.lru_map) {
			parent.device_memory_allocator->reset_pool(v.value);
		}

		/*
		for (auto& lra : impl->lra_recycle) {
			impl->cleanup_transient_bundle_recursively(lra);
		}
		impl->lra_recycle.clear();
		*/

		impl->descriptor_sets.collect(parent, frames_in_flight * 2);
		impl->scratch_buffers.collect(parent, frames_in_flight * 2);
	}

	FrameAllocator::~FrameAllocator() {
	}

	void FrameAllocator::deallocate(Image i) {
		std::lock_guard _(impl->recycle_lock);
		impl->image_recycle.push_back(i);
	}

	void FrameAllocator::deallocate(ImageView iv) {
		std::lock_guard _(impl->recycle_lock);
		impl->image_view_recycle.push_back(iv.payload);
	}

	void FrameAllocator::deallocate(VkPipeline p) {
		std::lock_guard _(impl->recycle_lock);
		impl->pipeline_recycle.push_back(p);
	}

	void FrameAllocator::deallocate(Buffer b) {
		std::lock_guard _(impl->recycle_lock);
		impl->buffer_recycle.push_back(b);
	}

	void FrameAllocator::deallocate(PersistentDescriptorSet b) {
		std::lock_guard _(impl->recycle_lock);
		impl->pds_recycle.push_back(std::move(b));
	}

	void FrameAllocator::deallocate(std::vector<vuk::Image>&& images) {
		std::lock_guard _(impl->recycle_lock);
		impl->image_recycle.insert(impl->image_recycle.end(), images.begin(), images.end());
	}

	void FrameAllocator::deallocate(std::vector<VkImageView>&& images) {
		std::lock_guard _(impl->recycle_lock);
		impl->image_view_recycle.insert(impl->image_view_recycle.end(), images.begin(), images.end());
	}

	/*void FrameAllocator::deallocate(std::vector<LinearResourceAllocator*>&& lras) {
		std::lock_guard _(impl->recycle_lock);
		impl->lra_recycle.insert(impl->lra_recycle.end(), lras.begin(), lras.end());
	}*/

	void FrameAllocator::deallocate(DescriptorSet ds) {
		std::lock_guard _(impl->recycle_lock);
		// note that since we collect at integer times FC, we are releasing the DS back to the right pool
		impl->pool_cache.acquire(ds.layout_info, absolute_frame).free_sets.enqueue(ds.descriptor_set);
	}

	void FrameAllocator::destroy(Token t) {
		parent.destroy(t);
	}
	TokenData& FrameAllocator::get_token_data(Token t) {
		return parent.get_token_data(t);
	}
	VkSemaphore FrameAllocator::allocate_timeline_semaphore(uint64_t initial_value, uint64_t absolute_frame, SourceLocation) {
		return parent.allocate_timeline_semaphore(initial_value, absolute_frame, VUK_HERE());
	}


	// -------------------- TL
	struct ThreadLocalFrameAllocatorImpl {
		ThreadLocalFrameAllocatorImpl(ThreadLocalFrameAllocator& tlfa) :
			commandbuffer_pool(tlfa.parent.impl->commandbuffer_pools),
			semaphore_pool(tlfa.parent.impl->semaphore_pools),
			fence_pool(tlfa.parent.impl->fence_pools),
			tsquery_pool(tlfa.parent.impl->tsquery_pools),
			sampled_images(tlfa.parent.impl->sampled_images) {
		}

		PTPoolView<VkCommandBuffer> commandbuffer_pool;
		PTPoolView<VkSemaphore> semaphore_pool;
		PTPoolView<VkFence> fence_pool;
		PTPoolView<TimestampQuery> tsquery_pool;
		PTPoolView<SampledImage> sampled_images;

		// recycling global objects
		std::vector<Buffer> buffer_recycle;
		std::vector<vuk::Image> image_recycle;
		std::vector<VkImageView> image_view_recycle;
		//std::vector<LinearResourceAllocator*> linear_allocators;
	};

	ThreadLocalFrameAllocator::ThreadLocalFrameAllocator(FrameAllocator& fa, unsigned tid) : Allocator(fa.ctx), tid(tid), parent(fa) {
		impl = new ThreadLocalFrameAllocatorImpl(*this);
	}

	void ThreadLocalFrameAllocator::deallocate(vuk::DescriptorSet ds) {
		parent.deallocate(ds);
	}

	void ThreadLocalFrameAllocator::deallocate(vuk::Image image) {
		impl->image_recycle.push_back(image);
	}

	void ThreadLocalFrameAllocator::deallocate(vuk::ImageView image) {
		impl->image_view_recycle.push_back(image.payload);
	}

	void ThreadLocalFrameAllocator::destroy(Token t) { parent.parent.destroy(t); }
	TokenData& ThreadLocalFrameAllocator::get_token_data(Token t) { return parent.parent.get_token_data(t); }

	TimestampQuery ThreadLocalFrameAllocator::register_timestamp_query(Query) {
		assert(0);
		return TimestampQuery();
	}

	Buffer ThreadLocalFrameAllocator::allocate_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, size_t size, size_t alignment) {
		bool create_mapped = mem_usage == MemoryUsage::eCPUonly || mem_usage == MemoryUsage::eCPUtoGPU || mem_usage == MemoryUsage::eGPUtoCPU;
		PoolSelect ps{ mem_usage, buffer_usage };
		auto& pool = parent.impl->scratch_buffers.acquire(parent.parent, ps, tid);
		return parent.parent.device_memory_allocator->allocate_buffer(pool, size, alignment, create_mapped);
	}

	Buffer ThreadLocalFrameAllocator::create_buffer(MemoryUsage mem_usage, vuk::BufferUsageFlags buffer_usage, void* data, size_t size, size_t alignment) {
		assert(mem_usage == MemoryUsage::eCPUonly || mem_usage == MemoryUsage::eCPUtoGPU || mem_usage == MemoryUsage::eGPUtoCPU);
		auto buffer = allocate_buffer(mem_usage, buffer_usage, size, alignment);
		auto tok = vuk::copy_to_buffer(*this, vuk::Domain::eAny, buffer, data, size);
		destroy(tok);
		return buffer;
	}

	Token ThreadLocalFrameAllocator::allocate_token() {
		return ctx.create_token();
	}

	struct RGImage ThreadLocalFrameAllocator::allocate_rendertarget(const struct RGCI& rgci) {
		return parent.parent.allocate_rendertarget(rgci, parent.absolute_frame, VUK_HERE());
	}

	VkCommandBuffer ThreadLocalFrameAllocator::allocate_command_buffer(VkCommandBufferLevel level, uint32_t queue_family_index) {
		CommandBufferCreateInfo cbi{ level, queue_family_index };
		// TODO: queue_family_index? -> multipool
		return impl->commandbuffer_pool.allocate(level, 1)[0];
	}
	ImageView ThreadLocalFrameAllocator::allocate_image_view(const ImageViewCreateInfo& i) {
		assert(0);
		return parent.parent.allocate_image_view(i, parent.absolute_frame, VUK_HERE()).release();
	}
	Sampler ThreadLocalFrameAllocator::allocate_sampler(const SamplerCreateInfo& s) {
		return parent.parent.allocate_sampler(s, parent.absolute_frame, VUK_HERE());
	}
	DescriptorSet ThreadLocalFrameAllocator::allocate_descriptorset(const SetBinding& s) {
		return parent.parent.allocate_descriptorset(s, parent.absolute_frame, VUK_HERE());
	}
	PipelineInfo ThreadLocalFrameAllocator::allocate_pipeline(const PipelineInstanceCreateInfo& d) {
		return parent.parent.allocate_pipeline(d, parent.absolute_frame, VUK_HERE());
	}
	VkFramebuffer ThreadLocalFrameAllocator::allocate_framebuffer(const struct FramebufferCreateInfo& d) {
		return parent.parent.allocate_framebuffer(d, parent.absolute_frame, VUK_HERE());
	}
	VkRenderPass ThreadLocalFrameAllocator::allocate_renderpass(const struct RenderPassCreateInfo& d) {
		return parent.parent.allocate_renderpass(d, parent.absolute_frame, VUK_HERE());
	}
	VkSemaphore ThreadLocalFrameAllocator::allocate_semaphore() {
		return parent.parent.allocate_semaphore(parent.absolute_frame, VUK_HERE());
	}
	VkFence ThreadLocalFrameAllocator::allocate_fence() {
		return parent.parent.allocate_fence(parent.absolute_frame, VUK_HERE());
	}
	VkSemaphore ThreadLocalFrameAllocator::allocate_timeline_semaphore(uint64_t initial_value, uint64_t absolute_frame, SourceLocation) {
		return parent.parent.allocate_timeline_semaphore(initial_value, parent.absolute_frame, VUK_HERE());
	}
}