#include "Cache.hpp"
#include "vuk/Context.hpp"
#include "LegacyGPUAllocator.hpp"
#include "vuk/PipelineInstance.hpp"

namespace vuk {
	template<class T>
	T& Cache<T>::acquire(const create_info_t<T>& ci) {
		assert(0);
		static T t;
		return t;
	}

	template<class T>
	T& Cache<T>::acquire(const create_info_t<T>& ci, uint64_t current_frame) {
		std::shared_lock _(cache_mtx);
		if (auto it = lru_map.find(ci); it != lru_map.end()) {
			it->second.last_use_frame = current_frame;
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(cache_mtx);
			auto pit = pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, current_frame };
			it = lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}

	template<class T>
	void Cache<T>::collect(uint64_t current_frame, size_t threshold) {
		std::unique_lock _(cache_mtx);
		for (auto it = lru_map.begin(); it != lru_map.end();) {
			auto last_use_frame = it->second.last_use_frame;
			if ((int64_t)current_frame - (int64_t)last_use_frame > (int64_t)threshold) {
				ctx.destroy(*it->second.ptr);
				pool.erase(pool.get_iterator_from_pointer(it->second.ptr));
				it = lru_map.erase(it);
			} else {
				++it;
			}
		}
	}

	template<>
	ShaderModule& Cache<ShaderModule>::acquire(const create_info_t<ShaderModule>& ci) {
		std::shared_lock _(cache_mtx);
		if (auto it = lru_map.find(ci); it != lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(cache_mtx);
			auto pit = pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}

	template<>
	PipelineBaseInfo& Cache<PipelineBaseInfo>::acquire(const create_info_t<PipelineBaseInfo>& ci) {
		std::shared_lock _(cache_mtx);
		if (auto it = lru_map.find(ci); it != lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(cache_mtx);
			auto pit = pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}

	template<>
	ComputePipelineBaseInfo& Cache<ComputePipelineBaseInfo>::acquire(const create_info_t<ComputePipelineBaseInfo>& ci) {
		std::shared_lock _(cache_mtx);
		if (auto it = lru_map.find(ci); it != lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(cache_mtx);
			auto pit = pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}


	template<>
	DescriptorSetLayoutAllocInfo& Cache<DescriptorSetLayoutAllocInfo>::acquire(const create_info_t<DescriptorSetLayoutAllocInfo>& ci) {
		std::shared_lock _(cache_mtx);
		if (auto it = lru_map.find(ci); it != lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(cache_mtx);
			auto pit = pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}

	template<>
	VkPipelineLayout& Cache<VkPipelineLayout>::acquire(const create_info_t<VkPipelineLayout>& ci) {
		std::shared_lock _(cache_mtx);
		if (auto it = lru_map.find(ci); it != lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(cache_mtx);
			auto pit = pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}
	// unfortunately, we need to manage extended_data lifetime here
	template<>
	PipelineInfo& Cache<PipelineInfo>::acquire(const create_info_t<PipelineInfo>& ci, uint64_t current_frame) {
		std::shared_lock _(cache_mtx);
		if (auto it = lru_map.find(ci); it != lru_map.end()) {
			it->second.last_use_frame = current_frame;
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(cache_mtx);
			auto ci_copy = ci;
			if (!ci_copy.is_inline()) {
				ci_copy.extended_data = new std::byte[ci_copy.extended_size];
				memcpy(ci_copy.extended_data, ci.extended_data, ci_copy.extended_size);
			}
			auto pit = pool.emplace(ctx.create(ci_copy));
			typename Cache::LRUEntry entry{ &*pit, current_frame };
			it = lru_map.emplace(ci_copy, entry).first;
			return *it->second.ptr;
		}
	}

	template<>
	void Cache<PipelineInfo>::collect(uint64_t current_frame, size_t threshold) {
		std::unique_lock _(cache_mtx);
		for (auto it = lru_map.begin(); it != lru_map.end();) {
			auto last_use_frame = it->second.last_use_frame;
			if ((int64_t)current_frame - (int64_t)last_use_frame > (int64_t)threshold) {
				ctx.destroy(*it->second.ptr);
				if (!it->first.is_inline()) {
					delete it->first.extended_data;
				}
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
			ctx.destroy(v);
		}
	}

	template class Cache<vuk::PipelineInfo>;
	template class Cache<vuk::PipelineBaseInfo>;
	template class Cache<vuk::ComputePipelineInfo>;
	template class Cache<vuk::ComputePipelineBaseInfo>;
	template class Cache<VkRenderPass>;
	template class Cache<vuk::Sampler>;
	template class Cache<VkPipelineLayout>;
	template class Cache<vuk::DescriptorSetLayoutAllocInfo>;
	template class Cache<vuk::ShaderModule>;
	template class Cache<vuk::RGImage>;

	template<class T, size_t FC>
	PerFrameCache<T, FC>::~PerFrameCache() {
		for (auto& p : _data) {
			for (auto& [k, v] : p.lru_map) {
				ctx.destroy(v.value);
			}
		}
	}

	template<class T, size_t FC>
	T& PerFrameCache<T, FC>::acquire(const create_info_t<T>& ci, uint64_t current_frame) {
		auto& data = _data[current_frame % FC];
		if (auto it = data.lru_map.find(ci); it != data.lru_map.end()) {
			it->second.last_use_frame = current_frame;
			return it->second.value;
		} else {
			// if the value is not in the cache, we look in our per thread buffers
			// if it doesn't exist there either, we add it
			auto& ptv = data.per_thread_append_v[0 /*ptc.tid*/];
			auto& ptk = data.per_thread_append_k[0 /*ptc.tid*/]; // TODO: restore TIDs
			auto pit = std::find(ptk.begin(), ptk.end(), ci);
			if (pit == ptk.end()) {
				ptv.emplace_back(ctx.create(ci));
				pit = ptk.insert(ptk.end(), ci);
			}
			auto index = std::distance(ptk.begin(), pit);
			return ptv[index];
		}
	}

	template<class T, size_t FC>
	void PerFrameCache<T, FC>::collect(uint64_t current_frame, size_t threshold) {
		auto local_frame = current_frame % FC;
		auto& data = _data[local_frame];
		std::unique_lock _(data.cache_mtx);
		for (auto it = data.lru_map.begin(); it != data.lru_map.end();) {
			if (current_frame - it->second.last_use_frame > threshold) {
				ctx.destroy(it->second.value);
				it = data.lru_map.erase(it);
			} else {
				++it;
			}
		}

		for (size_t tid = 0; tid < _data[local_frame].per_thread_append_v.size(); tid++) {
			auto& vs = _data[local_frame].per_thread_append_v[tid];
			auto& ks = _data[local_frame].per_thread_append_k[tid];
			for (size_t i = 0; i < vs.size(); i++) {
				if (data.lru_map.find(ks[i]) == data.lru_map.end()) {
					data.lru_map.emplace(ks[i], LRUEntry{ std::move(vs[i]), current_frame });
				} else {
					ctx.destroy(vs[i]);
				}
			}
			vs.clear();
			ks.clear();
		}
	}

	template class PerFrameCache<vuk::DescriptorSet, Context::FC>;
	template class PerFrameCache<LegacyLinearAllocator, Context::FC>;

	template class Cache<vuk::DescriptorPool>;

	void DescriptorPool::grow(Context& ctx, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info) {
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
		vkCreateDescriptorPool(ctx.device, &dpci, nullptr, &pool);
		pools.emplace_back(pool);

		VkDescriptorSetAllocateInfo dsai{ .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
		dsai.descriptorPool = pools.back();
		dsai.descriptorSetCount = dpci.maxSets;
		std::vector<VkDescriptorSetLayout> layouts(dpci.maxSets, layout_alloc_info.layout);
		dsai.pSetLayouts = layouts.data();
		// allocate all the descriptorsets
		std::vector<VkDescriptorSet> sets(dsai.descriptorSetCount);
		vkAllocateDescriptorSets(ctx.device, &dsai, sets.data());
		free_sets.enqueue_bulk(sets.data(), sets.size());
		sets_allocated = dpci.maxSets;

		grow_mutex.unlock();
	}

	VkDescriptorSet DescriptorPool::acquire(Context& ctx, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info) {
		VkDescriptorSet ds;
		while (!free_sets.try_dequeue(ds)) {
			grow(ctx, layout_alloc_info);
		}
		return ds;
	}
}