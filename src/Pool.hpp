#pragma once

#include <span>
#include <plf_colony.h>
#include <mutex>
#include <vector>
#include <vuk/vuk_fwd.hpp>
#include <vuk/Config.hpp>

namespace vuk {
	struct GlobalAllocator;

	template<class T>
	struct PooledType {
		std::vector<T> values;
		size_t needle = 0;

		PooledType(GlobalAllocator&) {}
		std::span<T> acquire(GlobalAllocator&, size_t count);
		void reset(GlobalAllocator&) { needle = 0; }
		void free(GlobalAllocator&);
	};

	template<>
	void PooledType<VkFence>::reset(GlobalAllocator&);

	template<>
	struct PooledType<VkCommandBuffer> {
		VkCommandPool pool;
		std::vector<VkCommandBuffer> p_values;
		std::vector<VkCommandBuffer> s_values;
		size_t p_needle = 0;
		size_t s_needle = 0;

		PooledType(GlobalAllocator&);
		std::span<VkCommandBuffer> acquire(GlobalAllocator&, VkCommandBufferLevel, size_t count);
		void reset(GlobalAllocator&);
		void free(GlobalAllocator&);
	};

	struct TimestampQuery;

	template<>
	struct PooledType<TimestampQuery> {
		VkQueryPool pool;
		std::vector<TimestampQuery> values;
		std::vector<uint64_t> host_values;
		std::vector<std::pair<uint64_t, uint64_t>> id_to_value_mapping;
		size_t needle = 0;

		PooledType(GlobalAllocator&);
		std::span<TimestampQuery> acquire(GlobalAllocator&, size_t count);
		void get_results(GlobalAllocator&);
		void reset(GlobalAllocator&);
		void free(GlobalAllocator&);
	};

	template<class T>
	struct Pool {
		std::mutex lock;
		plf::colony<PooledType<T>> store;
		std::vector<plf::colony<PooledType<T>>> per_frame_storage;
		GlobalAllocator& ga;

		Pool(GlobalAllocator& ga, size_t FC) : per_frame_storage(FC), ga(ga) {}

		PooledType<T>* acquire_one_into(plf::colony<PooledType<T>>& dst) {
			std::lock_guard _(lock);
			if (!store.empty()) {
				auto& last_elem = *(--store.end());
				auto new_it = dst.emplace(std::move(last_elem));
				store.erase(--store.end());
				return &*new_it;
			} else {
				return &*dst.emplace(PooledType<T>(ga));
			}
		}

		void reset(unsigned frame) {
			std::lock_guard _(lock);
			for (auto& t : per_frame_storage[frame]) {
				t.reset(ga);
			}
			store.splice(per_frame_storage[frame]);
		}

		~Pool() {
			// return all to pool
			for (auto& pf : per_frame_storage) {
				for (auto& s : pf) {
					s.free(ga);
				}
			}
			for (auto& s : store) {
				s.free(ga);
			}
		}
	};

	template<class T>
	struct PoolView {
		std::mutex lock;
		GlobalAllocator& ga;
		plf::colony<PooledType<T>>& frame_values;
		unsigned frame;
		Pool<T>& storage;
		
		PoolView(Pool<T>& st, unsigned frame) : ga(st.ga), frame_values(st.per_frame_storage[frame]), frame(frame), storage(st) {}

		void reset() {
			storage.reset(frame);
		}

		PooledType<T>& allocate_thread() {
			std::lock_guard _(lock);
			return *storage.acquire_one_into(frame_values);
		};
	};

	template<class T>
	struct PTPoolView {
		PooledType<T>& pool;
		GlobalAllocator& ga;

		PTPoolView(PoolView<T>& pool_view) : pool(pool_view.allocate_thread()), ga(pool_view.ga) {}

		template<class... Args>
		decltype(auto) allocate(Args&&... args) {
			return pool.acquire(ga, std::forward<Args>(args)...);
		}

		void deallocate(T value) {
			return pool.release(value);
		}
	};
}
