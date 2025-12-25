#include "vuk/runtime/Cache.hpp"
#include "vuk/runtime/vk/PipelineInstance.hpp"

#include <mutex>
#include <plf_colony.h>
#include <robin_hood.h>
#include <shared_mutex>

namespace vuk {
	template<class T>
	struct CacheImpl {
		plf::colony<T> pool;
		robin_hood::unordered_node_map<create_info_t<T>, typename Cache<T>::LRUEntry> lru_map;
		std::shared_mutex cache_mtx;
	};

	template<class T>
	Cache<T>::Cache(void* allocator, create_fn create, destroy_fn destroy) : impl(new CacheImpl<T>()), create(create), destroy(destroy), allocator(allocator) {}

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
			auto pit = impl->pool.emplace(create(allocator, ci));
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
				destroy(allocator, *it->second.ptr);
				impl->pool.erase(impl->pool.get_iterator(it->second.ptr));
				it = impl->lru_map.erase(it);
			} else {
				++it;
			}
		}
	}

	template<class T>
	void Cache<T>::clear() {
		std::unique_lock _(impl->cache_mtx);
		for (auto it = impl->pool.begin(); it != impl->pool.end(); ++it) {
			destroy(allocator, *it);
		}
		impl->pool.clear();
		impl->lru_map.clear();
	}

	template<>
	vuk::ShaderModule& Cache<vuk::ShaderModule>::acquire(const create_info_t<vuk::ShaderModule>& ci) {
		std::shared_lock _(impl->cache_mtx);
		if (auto it = impl->lru_map.find(ci); it != impl->lru_map.end()) {
			if (it->second.load_cnt.load(std::memory_order_relaxed) == 0) { // perform a relaxed load to skip the atomic_wait path
				std::atomic_wait(&it->second.load_cnt, 0);
			}
			return *it->second.ptr;
		} else {
			_.unlock();
			auto elem = create(allocator, ci);
			std::unique_lock ulock(impl->cache_mtx);
			typename Cache::LRUEntry entry{ nullptr, INT64_MAX };
			it = impl->lru_map.emplace(ci, entry).first;
			auto pit = impl->pool.emplace(std::move(elem));
			it->second.ptr = &*pit;
			it->second.load_cnt.store(1);
			it->second.load_cnt.notify_all();
			return *it->second.ptr;
		}
	}

	template<>
	vuk::PipelineBaseInfo& Cache<vuk::PipelineBaseInfo>::acquire(const create_info_t<vuk::PipelineBaseInfo>& ci) {
		std::shared_lock _(impl->cache_mtx);
		if (auto it = impl->lru_map.find(ci); it != impl->lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			auto elem = create(allocator, ci);
			std::unique_lock ulock(impl->cache_mtx);
			auto pit = impl->pool.emplace(std::move(elem));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = impl->lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}

	template<>
	vuk::DescriptorSetLayoutAllocInfo& Cache<vuk::DescriptorSetLayoutAllocInfo>::acquire(const create_info_t<vuk::DescriptorSetLayoutAllocInfo>& ci) {
		std::shared_lock _(impl->cache_mtx);
		if (auto it = impl->lru_map.find(ci); it != impl->lru_map.end()) {
			return *it->second.ptr;
		} else {
			_.unlock();
			std::unique_lock ulock(impl->cache_mtx);
			auto pit = impl->pool.emplace(create(allocator, ci));
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
			auto pit = impl->pool.emplace(create(allocator, ci));
			typename Cache::LRUEntry entry{ &*pit, INT64_MAX };
			it = impl->lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}
	// unfortunately, we need to manage extended_data lifetime here
	template<>
	vuk::GraphicsPipelineInfo& Cache<vuk::GraphicsPipelineInfo>::acquire(const create_info_t<vuk::GraphicsPipelineInfo>& ci, uint64_t current_frame) {
		std::shared_lock _(impl->cache_mtx);
		if (auto it = impl->lru_map.find(ci); it != impl->lru_map.end()) {
			it->second.last_use_frame = current_frame;
			if (it->second.load_cnt.load(std::memory_order_relaxed) == 0) { // perform a relaxed load to skip the atomic_wait path
				std::atomic_wait(&it->second.load_cnt, 0);
			}
			return *it->second.ptr;
		} else {
			_.unlock();
			auto ci_copy = ci;
			if (!ci_copy.is_inline()) {
				ci_copy.extended_data = new std::byte[ci_copy.extended_size];
				memcpy(ci_copy.extended_data, ci.extended_data, ci_copy.extended_size);
			}
			std::unique_lock ulock(impl->cache_mtx);
			typename Cache::LRUEntry entry{ nullptr, current_frame };
			it = impl->lru_map.emplace(ci_copy, entry).first;
			auto pit = impl->pool.emplace(create(allocator, ci_copy));
			ulock.unlock();
			it->second.ptr = &*pit;
			it->second.load_cnt.store(1);
			it->second.load_cnt.notify_all();
			return *it->second.ptr;
		}
	}

	template<>
	void Cache<vuk::GraphicsPipelineInfo>::collect(uint64_t current_frame, size_t threshold) {
		std::unique_lock _(impl->cache_mtx);
		for (auto it = impl->lru_map.begin(); it != impl->lru_map.end();) {
			auto last_use_frame = it->second.last_use_frame;
			if ((int64_t)current_frame - (int64_t)last_use_frame > (int64_t)threshold) {
				destroy(allocator, *it->second.ptr);
				if (!it->first.is_inline()) {
					delete it->first.extended_data;
				}
				impl->pool.erase(impl->pool.get_iterator(it->second.ptr));
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
			impl->pool.erase(impl->pool.get_iterator(it->second.ptr));
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
				impl->pool.erase(impl->pool.get_iterator(it->second.ptr));
				impl->lru_map.erase(it);
				return;
			}
		}
	}

	template<class T>
	Cache<T>::~Cache() {
		for (auto& v : impl->pool) {
			destroy(allocator, v);
		}
		delete impl;
	}

	template class Cache<vuk::GraphicsPipelineInfo>;
	template class Cache<vuk::PipelineBaseInfo>;
	template class Cache<vuk::ComputePipelineInfo>;
	template class Cache<vuk::RayTracingPipelineInfo>;
	template class Cache<VkRenderPass>;
	template class Cache<vuk::Sampler>;
	template class Cache<VkPipelineLayout>;
	template class Cache<vuk::DescriptorSetLayoutAllocInfo>;
	template class Cache<vuk::ShaderModule>;
	template struct CacheImpl<vuk::ShaderModule>;
	template class Cache<vuk::ImageWithIdentity>;
	template class Cache<vuk::ImageView<>>;

	template class Cache<vuk::DescriptorPool>;
} // namespace vuk