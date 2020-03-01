#include "Cache.hpp"
#include "Context.hpp"

namespace vuk {
	template<class T>
	T& Cache<T>::PFPTView::acquire(const create_info_t<T>& ci) {
		auto& cache = view.cache;
		std::shared_lock _(cache.cache_mtx);
		if (auto it = cache.lru_map.find(ci); it != cache.lru_map.end()) {
			it->second.last_use_frame = ptc.ifc.frame;
			return *it->second.ptr;
		}
		else {
			_.unlock();
			std::unique_lock ulock(cache.cache_mtx);
			auto pit = cache.pool.emplace(create<T>(ptc, ci));
			typename Cache::LRUEntry entry{ &*pit, ptc.ifc.frame };
			it = cache.lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}
	template<class T>
	void Cache<T>::PFView::collect(size_t threshold) {
		for (auto& v : cache.lru_map) {
			if (ifc.frame - v.second.last_use_frame > threshold) {
				// TODO: recycling DescriptorSets
				if constexpr (!std::is_same_v<T, vk::DescriptorSet> && !std::is_same_v<T, vuk::PipelineInfo>)
					ifc.ctx.device.destroy(*v.second.ptr);
				cache.pool.erase(cache.pool.get_iterator_from_pointer(v.second.ptr));
				cache.lru_map.erase(v.first);
			}
		}
	}
	
	template<class T>
	Cache<T>::~Cache() {
		for (auto& v : pool) {
			if constexpr (!std::is_same_v<T, vk::DescriptorSet> && !std::is_same_v<T, vuk::PipelineInfo>)
				ctx.device.destroy(v);
		}
	}

	template struct Cache<vuk::PipelineInfo>;
	template struct Cache<vk::RenderPass>;
	template struct Cache<vk::DescriptorSet>;

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
			it->second.last_use_frame = ptc.ifc.frame;
			return *it->second.ptr;
		}
		else {
			_.unlock();
			std::unique_lock ulock(data.cache_mtx);
			auto pit = data.pool.emplace(create<T>(ptc, ci));
			typename PerFrameCache::LRUEntry entry{ &*pit, ptc.ifc.frame };
			it = data.lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}
	template<class T, size_t FC>
	void PerFrameCache<T, FC>::PFView::collect(size_t threshold) {
		auto& data = cache.data[ifc.frame];
		for (auto& v : data.lru_map) {
			if (ifc.frame - v.second.last_use_frame > threshold) {
				ifc.ctx.destroy(*v.second.ptr);
				data.pool.erase(data.pool.get_iterator_from_pointer(v.second.ptr));
				data.lru_map.erase(v.first);
			}
		}
	}

	template struct PerFrameCache<vuk::RGImage, Context::FC>;
	template struct PerFrameCache<Allocator::Pool, Context::FC>;
}