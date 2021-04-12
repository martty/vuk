#include <Cache.hpp>
#include <vuk/Context.hpp>
#include <Allocator.hpp>
#include <vuk/GlobalAllocator.hpp>
#include <vuk/Descriptor.hpp>

namespace vuk {
	template<class T>
	T& Cache<T>::acquire(const create_info_t<T>& ci, uint64_t current_frame){
		std::shared_lock _(cache_mtx);
		if (auto it = lru_map.find(ci); it != lru_map.end()) {
			it->second.last_use_frame = current_frame;
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(cache_mtx);
			auto pit = pool.emplace(ga.create(ci, current_frame));
			typename Cache::LRUEntry entry{ &*pit, current_frame };
			it = lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}

	template<class T>
	void Cache<T>::collect(uint64_t current_frame, uint64_t threshold) {
		std::unique_lock _(cache_mtx);
		for (auto it = lru_map.begin(); it != lru_map.end();) {
			if (current_frame - it->second.last_use_frame > threshold) {
				ga.destroy(*it->second.ptr);
				pool.erase(pool.get_iterator_from_pointer(it->second.ptr));
				it = lru_map.erase(it);
			} else {
				++it;
			}
		}
	}

	template<class T>
	Cache<T>::~Cache() {
		for (auto& v : pool) {
			ga.destroy(v);
		}
	}

	template class Cache<vuk::PipelineInfo>;
	template class Cache<vuk::PipelineBaseInfo>;
	template class Cache<vuk::ComputePipelineInfo>;
	template class Cache<VkRenderPass>;
	template class Cache<VkFramebuffer>;
	template class Cache<vuk::Sampler>;
	template class Cache<VkPipelineLayout>;
	template class Cache<vuk::DescriptorSetLayoutAllocInfo>;
	template class Cache<vuk::ShaderModule>;
	template class Cache<vuk::RGImage>;

	template<class T>
	PerFrameCache<T>::PerFrameCache(GlobalAllocator& ga, size_t FC) : FC(FC), ga(ga), data(new PFCPerFrame<T>[FC]) { }

	template<class T>
	PerFrameCache<T>::~PerFrameCache() {
		for (auto& p : std::span(data.get(), FC)) {
			for (auto& [k, v] : p.lru_map) {
				ga.destroy(v.value);
			}
		}
	}

	template<class T>
	T& PerFrameCacheView<T>::acquire(GlobalAllocator& ga, const create_info_t<T>& ci, unsigned tid) {
		auto& data = cache;
		if (auto it = data.lru_map.find(ci); it != data.lru_map.end()) {
			it->second.last_use_frame = absolute_frame;
			return it->second.value;
		} else {
			// if the value is not in the cache, we look in our per thread buffers
			// if it doesn't exist there either, we add it
			auto& ptv = data.per_thread_append_v[tid];
			auto& ptk = data.per_thread_append_k[tid];
			auto pit = std::find(ptk.begin(), ptk.end(), ci);
			if (pit == ptk.end()) {
				ptv.emplace_back(ga.create(ci, absolute_frame));
				pit = ptk.insert(ptk.end(), ci);
			}
			auto index = std::distance(ptk.begin(), pit);
			return ptv[index];
		}
	}

	template<class T>
	void PerFrameCacheView<T>::collect(GlobalAllocator& ga, size_t threshold) {
		std::unique_lock _(cache.cache_mtx);
		for (auto it = cache.lru_map.begin(); it != cache.lru_map.end();) {
			if (absolute_frame - it->second.last_use_frame > threshold) {
				ga.destroy(it->second.value);
				it = cache.lru_map.erase(it);
			} else {
				++it;
			}
		}

		for (size_t tid = 0; tid < cache.per_thread_append_v.size(); tid++) {
			auto& vs = cache.per_thread_append_v[tid];
			auto& ks = cache.per_thread_append_k[tid];
			for (size_t i = 0; i < vs.size(); i++) {
				if (cache.lru_map.find(ks[i]) == cache.lru_map.end()) {
					cache.lru_map.emplace(ks[i], typename PFCPerFrame<T>::LRUEntry{ std::move(vs[i]), absolute_frame });
				} else {
					ga.destroy(vs[i]);
				}
			}
			vs.clear();
			ks.clear();
		}
	}

	template struct PerFrameCache<vuk::DescriptorSet>;
	template struct PerFrameCache<LinearAllocator>;

	template struct PerFrameCacheView<vuk::DescriptorSet>;
	template struct PerFrameCacheView<LinearAllocator>;

	template class Cache<vuk::DescriptorPool>;

	void DescriptorPool::grow(GlobalAllocator& ga, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info) {
		if (!grow_mutex.try_lock())
			return;
		VkDescriptorPoolCreateInfo dpci{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
		dpci.maxSets = sets_allocated == 0 ? 1 : sets_allocated * 2;
		std::array<VkDescriptorPoolSize, 12> descriptor_counts = {};
		uint32_t used_idx = 0;
		for (auto i = 0; i < descriptor_counts.size(); i++) {
			if (layout_alloc_info.descriptor_counts[i] > 0) {
				auto& d = descriptor_counts[used_idx];
				d.type = VkDescriptorType(i);
				d.descriptorCount = layout_alloc_info.descriptor_counts[i] * dpci.maxSets;
				used_idx++;
			}
		}
		dpci.pPoolSizes = descriptor_counts.data();
		dpci.poolSizeCount = used_idx;
		VkDescriptorPool pool;
		vkCreateDescriptorPool(ga.device, &dpci, nullptr, &pool);
		pools.emplace_back(pool);

		VkDescriptorSetAllocateInfo dsai{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		dsai.descriptorPool = pools.back();
		dsai.descriptorSetCount = dpci.maxSets;
		std::vector<VkDescriptorSetLayout> layouts(dpci.maxSets, layout_alloc_info.layout);
		dsai.pSetLayouts = layouts.data();
		// allocate all the descriptorsets
		std::vector<VkDescriptorSet> sets(dsai.descriptorSetCount);
		vkAllocateDescriptorSets(ga.device, &dsai, sets.data());
		free_sets.enqueue_bulk(sets.data(), sets.size());
		sets_allocated = dpci.maxSets;

		grow_mutex.unlock();
	}

	VkDescriptorSet DescriptorPool::acquire(GlobalAllocator& ga, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info) {
		VkDescriptorSet ds;
		while (!free_sets.try_dequeue(ds)) {
			grow(ga, layout_alloc_info);
		}
		return ds;
	}
}