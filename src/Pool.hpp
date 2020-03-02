#pragma once

#include <gsl/span>
#include <plf_colony.h>
#include <mutex>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace vuk {
	class Context;
	class InflightContext;
	class PerThreadContext;


	template<class T>
	struct PooledType {
		std::vector<T> values;
		size_t needle = 0;

		PooledType(Context&) {}
		gsl::span<T> acquire(PerThreadContext& ptc, size_t count);
		void reset(Context& ctx) { needle = 0; }
		void free(Context& ctx);
	};

	template<>
	void PooledType<vk::Fence>::reset(Context& ctx);

	template<>
	struct PooledType<vk::CommandBuffer> {
		vk::UniqueCommandPool pool;
		std::vector<vk::CommandBuffer> values;
		size_t needle = 0;

		PooledType(Context&);
		gsl::span<vk::CommandBuffer> acquire(PerThreadContext& ptc, size_t count);
		void reset(Context&);
		void free(Context&);
	};

	struct DescriptorSetLayoutAllocInfo;
	template<>
	struct PooledType<vk::DescriptorSet> {
		std::vector<vk::UniqueDescriptorPool> pools;
		size_t pool_needle = 0;
		std::vector<vk::DescriptorSet> sets;
		std::vector<vk::DescriptorSet> free_sets;
		size_t set_needle = 0;
		size_t sets_allocated = 0;

		PooledType(Context&);
		vk::DescriptorSet acquire(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);
		void reset(Context&);
		void free(Context&);

		vk::DescriptorPool get_pool(PerThreadContext& ptc, vuk::DescriptorSetLayoutAllocInfo layout_alloc_info);
	};

	template<class T, size_t FC>
	struct PFView;

	template<class T, size_t FC>
	struct Pool {
		std::mutex lock;
		plf::colony<PooledType<T>> store;
		std::array<plf::colony<PooledType<T>>, FC> per_frame_storage;
		Context& ctx;

		Pool(Context& ctx) : ctx(ctx) {	}

		PooledType<T>* acquire_one_into(plf::colony<PooledType<T>>& dst) {
			std::lock_guard _(lock);
			if (!store.empty()) {
				auto& last_elem = *(--store.end());
				auto new_it = dst.emplace(std::move(last_elem));
				store.erase(--store.end());
				return &*new_it;
			}
			else {
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
