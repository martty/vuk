#include "Cache.hpp"
#include "Context.hpp"

namespace vuk {
	template<class T>
	T Cache<T>::View::acquire(create_info_t<T> ci) {
		std::shared_lock _(cache.cache_mtx);
		if (auto it = cache.lru_map.find(ci); it != cache.lru_map.end()) {
			it->second.last_use_frame = ifc.frame;
			return *it->second.ptr;
		}
		else {
			_.unlock();
			std::unique_lock ulock(cache.cache_mtx);
			auto pit = cache.pool.emplace(create<T>(ifc.ctx, ci));
			typename Cache<T>::LRUEntry entry{ &*pit, ifc.frame };
			it = cache.lru_map.emplace(ci, entry).first;
			return *it->second.ptr;
		}
	}
	template<class T>
	void Cache<T>::View::collect(size_t threshold) {
		for (auto& v : cache.lru_map) {
			if (ifc.frame - v.second.last_use_frame > threshold) {
				ifc.ctx.device.destroy(*v.second.ptr);
				cache.pool.erase(cache.pool.get_iterator_from_pointer(v.second.ptr));
				cache.lru_map.erase(v.first);
			}
		}
	}
	
	template<class T>
	Cache<T>::~Cache() {
		for (auto& v : pool) {
			ctx.device.destroy(v);
		}
	}

	template struct Cache<vk::Pipeline>;
	template struct Cache<vk::RenderPass>;
}