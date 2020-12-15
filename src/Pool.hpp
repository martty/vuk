#pragma once

#include <span>
#include <plf_colony.h>
#include <mutex>
#include <vector>
#include <vuk/vuk_fwd.hpp>
#include <vuk/Config.hpp>

namespace vuk {
	template<class T>
	struct PooledType {
		std::vector<T> values;
		size_t needle = 0;

		PooledType(Context&) {}
		std::span<T> acquire(PerThreadContext& ptc, size_t count);
		void reset(Context& ctx) { needle = 0; }
		void free(Context& ctx);
	};

	template<>
	void PooledType<VkFence>::reset(Context& ctx);
	template<>
	void PooledType<VkEvent>::reset(Context& ctx);

	template<>
	struct PooledType<VkCommandBuffer> {
		VkCommandPool pool;
		std::vector<VkCommandBuffer> p_values;
		std::vector<VkCommandBuffer> s_values;
		size_t p_needle = 0;
		size_t s_needle = 0;

		PooledType(Context&);
		std::span<VkCommandBuffer> acquire(PerThreadContext& ptc, VkCommandBufferLevel, size_t count);
		void reset(Context&);
		void free(Context&);
	};

	struct TimestampQuery;

	template<>
	struct PooledType<TimestampQuery> {
		VkQueryPool pool;
		std::vector<TimestampQuery> values;
		std::vector<uint64_t> host_values;
		std::vector<std::pair<uint64_t, uint64_t>> id_to_value_mapping;
		size_t needle = 0;

		PooledType(Context&);
		std::span<TimestampQuery> acquire(PerThreadContext& ptc, size_t count);
		void get_results(Context&);
		void reset(Context&);
		void free(Context&);
	};

	template<class T, size_t FC>
	struct PFView;

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
			} 			else {
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

		struct PFPTView {
			PerThreadContext& ptc;
			PooledType<T>& pool;

			PFPTView(PerThreadContext& ptc, PooledType<T>& pool) : ptc(ptc), pool(pool) {}

			template<class... Args>
			decltype(auto) acquire(Args&&... args) {
				return pool.acquire(ptc, std::forward<Args>(args)...);
			}

			void release(T value) {
				return pool.release(value);
			}
		};

		struct PFView {
			std::mutex lock;
			Pool& storage;
			InflightContext& ifc;
			plf::colony<PooledType<T>>& frame_values;

			PFView(InflightContext& ifc, Pool& storage, plf::colony<PooledType<T>>& fv);

			PFPTView get_view(PerThreadContext& ptc) {
				std::lock_guard _(lock);
				return { ptc, *storage.acquire_one_into(frame_values) };
			}
		};

		PFView get_view(InflightContext& ctx);
	};
}
