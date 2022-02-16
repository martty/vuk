#include "Cache.hpp"
#include "LegacyGPUAllocator.hpp"
#include "vuk/Context.hpp"
#include "vuk/PipelineInstance.hpp"

#include <plf_colony.h>
#include <robin_hood.h>
#include <shared_mutex>

namespace vuk {
	template<class T>
	struct CacheImpl {
		plf::colony<T> pool;
		robin_hood::unordered_map<create_info_t<T>, typename Cache<T>::LRUEntry> lru_map; // possibly vector_map or an intrusive map
		std::shared_mutex cache_mtx;
	};

	template<class T>
	Cache<T>::Cache(Context& ctx) : ctx(ctx), impl(new CacheImpl<T>()) {}

	template<class T>
	T& Cache<T>::acquire(const create_info_t<T>& ci) {
		assert(0);
		static T t;
		return t;
	}

	template<class T>
	T& Cache<T>::acquire(const create_info_t<T>& ci, uint64_t current_frame) {
		std::shared_lock _(impl->cache_mtx);
		if (auto it = impl->lru_map.find(ci); it != impl->lru_map.end()) {
			it->second.last_use_frame = current_frame;
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(impl->cache_mtx);
			auto pit = impl->pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, current_frame };
			it = impl->lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}

	template<class T>
	void Cache<T>::collect(uint64_t current_frame, size_t threshold) {
		std::unique_lock _(impl->cache_mtx);
		for (auto it = impl->lru_map.begin(); it != impl->lru_map.end();) {
			auto last_use_frame = it->second.last_use_frame;
			if ((int64_t)current_frame - (int64_t)last_use_frame > (int64_t)threshold) {
				ctx.destroy(*it->second.ptr);
				impl->pool.erase(impl->pool.get_iterator_from_pointer(it->second.ptr));
				it = impl->lru_map.erase(it);
			} else {
				++it;
			}
		}
	}

	template<>
	ShaderModule& Cache<ShaderModule>::acquire(const create_info_t<ShaderModule>& ci) {
		std::shared_lock _(impl->cache_mtx);
		if (auto it = impl->lru_map.find(ci); it != impl->lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(impl->cache_mtx);
			auto pit = impl->pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = impl->lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}

	template<>
	PipelineBaseInfo& Cache<PipelineBaseInfo>::acquire(const create_info_t<PipelineBaseInfo>& ci) {
		std::shared_lock _(impl->cache_mtx);
		if (auto it = impl->lru_map.find(ci); it != impl->lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(impl->cache_mtx);
			auto pit = impl->pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = impl->lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}

	template<>
	DescriptorSetLayoutAllocInfo& Cache<DescriptorSetLayoutAllocInfo>::acquire(const create_info_t<DescriptorSetLayoutAllocInfo>& ci) {
		std::shared_lock _(impl->cache_mtx);
		if (auto it = impl->lru_map.find(ci); it != impl->lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(impl->cache_mtx);
			auto pit = impl->pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = impl->lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}

	template<>
	VkPipelineLayout& Cache<VkPipelineLayout>::acquire(const create_info_t<VkPipelineLayout>& ci) {
		std::shared_lock _(impl->cache_mtx);
		if (auto it = impl->lru_map.find(ci); it != impl->lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(impl->cache_mtx);
			auto pit = impl->pool.emplace(ctx.create(ci));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = impl->lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}
	// unfortunately, we need to manage extended_data lifetime here
	template<>
	PipelineInfo& Cache<PipelineInfo>::acquire(const create_info_t<PipelineInfo>& ci, uint64_t current_frame) {
		std::shared_lock _(impl->cache_mtx);
		if (auto it = impl->lru_map.find(ci); it != impl->lru_map.end()) {
			it->second.last_use_frame = current_frame;
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(impl->cache_mtx);
			auto ci_copy = ci;
			if (!ci_copy.is_inline()) {
				ci_copy.extended_data = new std::byte[ci_copy.extended_size];
				memcpy(ci_copy.extended_data, ci.extended_data, ci_copy.extended_size);
			}
			auto pit = impl->pool.emplace(ctx.create(ci_copy));
			typename Cache::LRUEntry entry{ &*pit, current_frame };
			it = impl->lru_map.emplace(ci_copy, entry).first;
			return *it->second.ptr;
		}
	}

	template<>
	void Cache<PipelineInfo>::collect(uint64_t current_frame, size_t threshold) {
		std::unique_lock _(impl->cache_mtx);
		for (auto it = impl->lru_map.begin(); it != impl->lru_map.end();) {
			auto last_use_frame = it->second.last_use_frame;
			if ((int64_t)current_frame - (int64_t)last_use_frame > (int64_t)threshold) {
				ctx.destroy(*it->second.ptr);
				if (!it->first.is_inline()) {
					delete it->first.extended_data;
				}
				impl->pool.erase(impl->pool.get_iterator_from_pointer(it->second.ptr));
				it = impl->lru_map.erase(it);
			} else {
				++it;
			}
		}
	}

	template<class T>
	std::optional<T> Cache<T>::remove(const create_info_t<T>& ci) {
		std::unique_lock _(impl->cache_mtx);
		auto it = impl->lru_map.find(ci);
		if (it != impl->lru_map.end()) {
			auto res = std::move(*it->second.ptr);
			impl->pool.erase(impl->pool.get_iterator_from_pointer(it->second.ptr));
			impl->lru_map.erase(it);
			return res;
		}
		return {};
	}

	template<class T>
	void Cache<T>::remove_ptr(const T* ptr) {
		std::unique_lock _(impl->cache_mtx);
		for (auto it = impl->lru_map.begin(); it != impl->lru_map.end(); ++it) {
			if (ptr == it->second.ptr) {
				impl->pool.erase(impl->pool.get_iterator_from_pointer(it->second.ptr));
				impl->lru_map.erase(it);
				return;
			}
		}
	}

	template<class T>
	Cache<T>::~Cache() {
		for (auto& v : impl->pool) {
			ctx.destroy(v);
		}
		delete impl;
	}

	template class Cache<vuk::PipelineInfo>;
	template class Cache<vuk::PipelineBaseInfo>;
	template class Cache<vuk::ComputePipelineInfo>;
	template class Cache<VkRenderPass>;
	template class Cache<vuk::Sampler>;
	template class Cache<VkPipelineLayout>;
	template class Cache<vuk::DescriptorSetLayoutAllocInfo>;
	template class Cache<vuk::ShaderModule>;
	template struct CacheImpl<vuk::ShaderModule>;
	template class Cache<vuk::RGImage>;

	template class Cache<vuk::DescriptorPool>;
} // namespace vuk