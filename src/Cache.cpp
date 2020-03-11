#include "Cache.hpp"
#include "Context.hpp"

namespace vuk {
	template<class T>
	T& Cache<T>::PFPTView::acquire(const create_info_t<T>& ci) {
		auto& cache = view.cache;
		std::shared_lock _(cache.cache_mtx);
		if (auto it = cache.lru_map.find(ci); it != cache.lru_map.end()) {
			it->second.last_use_frame = ptc.ifc.absolute_frame;
			return *it->second.ptr;
		}
		else {
			_.unlock();
			std::unique_lock ulock(cache.cache_mtx);
			auto pit = cache.pool.emplace(create<T>(ptc, ci));
			typename Cache::LRUEntry entry{ &*pit, ptc.ifc.absolute_frame };
			it = cache.lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}
	template<class T>
	void Cache<T>::PFPTView::collect(size_t threshold) {
		auto& cache = view.cache;
		std::unique_lock _(cache.cache_mtx);
		for (auto it = cache.lru_map.begin(); it != cache.lru_map.end(); ) {
			if (ptc.ifc.absolute_frame - it->second.last_use_frame > threshold) {
				ptc.destroy(*it->second.ptr);
				cache.pool.erase(cache.pool.get_iterator_from_pointer(it->second.ptr));
				it = cache.lru_map.erase(it);
			} else {
				++it;
			}
		}
	}
	
	template<class T>
	Cache<T>::~Cache() {
		for (auto& v : pool) {
			ctx.destroy(v);
		}
	}

	template struct Cache<vuk::PipelineInfo>;
	template struct Cache<vk::RenderPass>;
	template struct Cache<vuk::DescriptorSet>;
	template struct Cache<vk::Framebuffer>;
	template struct Cache<vk::Sampler>;
	template struct Cache<vk::PipelineLayout>;
	template struct Cache<vuk::DescriptorSetLayoutAllocInfo>;
	template struct Cache<vuk::ShaderModule>;

	template<class T, size_t FC>
	PerFrameCache<T, FC>::~PerFrameCache() {
		for (auto& p : data) {
			for (auto& v : p.pool) {
				ctx.destroy(v);
			}
		}
	}

	template<class T, size_t FC>
	T& PerFrameCache<T, FC>::PFPTView::acquire(const create_info_t<T>& ci) {
		auto& cache = view.cache;
		auto& data = cache.data[ptc.ifc.frame];
		std::shared_lock _(data.cache_mtx);
		if (auto it = data.lru_map.find(ci); it != data.lru_map.end()) {
			it->second.last_use_frame = ptc.ifc.absolute_frame;
			return *it->second.ptr;
		}
		else {
			_.unlock();
			std::unique_lock ulock(data.cache_mtx);
			auto pit = data.pool.emplace(create<T>(ptc, ci));
			typename PerFrameCache::LRUEntry entry{ &*pit, ptc.ifc.absolute_frame };
			it = data.lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}
	template<class T, size_t FC>
	void PerFrameCache<T, FC>::PFPTView::collect(size_t threshold) {
		auto& data = view.cache.data[ptc.ifc.frame];
		std::unique_lock _(data.cache_mtx);
		for (auto& v : data.lru_map) {
			if (ptc.ifc.absolute_frame - v.second.last_use_frame > threshold) {
				ptc.destroy(*v.second.ptr);
				data.pool.erase(data.pool.get_iterator_from_pointer(v.second.ptr));
				data.lru_map.erase(v.first);
			}
		}
	}

	template struct PerFrameCache<vuk::RGImage, Context::FC>;
	template struct PerFrameCache<Allocator::Pool, Context::FC>;
	template struct PerFrameCache<vuk::DescriptorPool, Context::FC>;
	
	vk::DescriptorPool DescriptorPool::get_pool(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info) {
		if (pools.size() < (pool_needle + 1)) {
			vk::DescriptorPoolCreateInfo dpci;
			dpci.maxSets = sets_allocated == 0 ? 1 : sets_allocated * 2;
			std::array<vk::DescriptorPoolSize, VkDescriptorType::VK_DESCRIPTOR_TYPE_END_RANGE> descriptor_counts = {};
			size_t used_idx = 0;
			for (auto i = 0; i < descriptor_counts.size(); i++) {
				if (layout_alloc_info.descriptor_counts[i] > 0) {
					auto& d = descriptor_counts[used_idx];
					d.type = vk::DescriptorType(i);
					d.descriptorCount = layout_alloc_info.descriptor_counts[i] * dpci.maxSets;
					used_idx++;
				}
			}
			dpci.pPoolSizes = descriptor_counts.data();
			dpci.poolSizeCount = used_idx;
			pools.emplace_back(ptc.ifc.ctx.device.createDescriptorPool(dpci));
			sets_allocated = dpci.maxSets;
		}
		return pools[pool_needle];
	}
	
	vk::DescriptorSet DescriptorPool::acquire(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info) {
		if (free_sets.size() > 0) {
			vk::DescriptorSet ret = free_sets.back();
			free_sets.pop_back();
			return ret;
		}
		// allocate new ds
		// we find a descriptorpool that can still allocate
		// or create a new pool
		// typically we don't loop
		VkDescriptorSet ds;
		while (true) {
			vk::DescriptorSetAllocateInfo dsai;
			dsai.descriptorPool = get_pool(ptc, layout_alloc_info);
			dsai.descriptorSetCount = 1;
			dsai.pSetLayouts = &layout_alloc_info.layout;
			auto result = vkAllocateDescriptorSets(ptc.ctx.device, &(VkDescriptorSetAllocateInfo)dsai, &ds);
			if (result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTED_POOL) {
				pool_needle++;
			} else {
				assert(result == VK_SUCCESS);
				return vk::DescriptorSet(ds);
			}
		}
	}
}