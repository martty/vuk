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

	template<class T, size_t FC>
	struct Pool {
		std::mutex lock;
		plf::colony<PooledType<T>> store;
		std::array<plf::colony<PooledType<T>>, FC> per_frame_storage;
		Context& ctx;

		Pool(Context& ctx) : ctx(ctx) {}

		PooledType<T>* acquire_one_into(plf::colony<PooledType<T>>& dst) {
			std::lock_guard _(lock);
			if (!store.empty()) {
				auto& last_elem = *(--store.end());
				auto new_it = dst.emplace(std::move(last_elem));
				store.erase(--store.end());
				return &*new_it;
			} else {
				return &*dst.emplace(PooledType<T>(ctx));
			}
		}

		void reset(unsigned frame) {
			std::lock_guard _(lock);
			for (auto& t : per_frame_storage[frame]) {
				t.reset(ctx);
			}
			store.splice(per_frame_storage[frame]);
		}

		~Pool() {
			// return all to pool
			for (auto& pf : per_frame_storage) {
				for (auto& s : pf) {
					s.free(ctx);
				}
			}
			for (auto& s : store) {
				s.free(ctx);
			}
		}
	};

	template<class T>
	struct PoolView {
		std::mutex lock;
		plf::colony<PooledType<T>>& frame_values;
		unsigned frame;
		void* storage;
		PooledType<T>&(*allocate_fn)(void*, plf::colony<PooledType<T>>&);

		template<size_t FC>
		PoolView(Pool<T, FC>& st, plf::colony<PooledType<T>>& fv, unsigned frame) : frame(frame), storage(&st), frame_values(st.per_frame_storage[frame]) {
			allocate_fn = [](void* st, plf::colony<PooledType<T>>& fv) {
				auto& storage = *reinterpret_cast<Pool<T, FC>>(st);
				return *storage.acquire_one_into(fv);
			}
		}

		void reset() {
			st.reset(frame);
		}

		PooledType<T>& allocate_thread() {
			std::lock_guard _(lock);
			return *allocate_fn(storage, frame_values);
		}
	};

	template<class T>
	struct PTPoolView {
		PooledType<T>& pool;
		GlobalAllocator& global_allocator;

		PTPoolView(PoolView<T>& pool_view) : pool(pool_view.allocate_thread()) {}

		template<class... Args>
		decltype(auto) allocate(Args&&... args) {
			return pool.acquire(global_allocator, std::forward<Args>(args)...);
		}

		void deallocate(T value) {
			return pool.release(value);
		}
	};
}
