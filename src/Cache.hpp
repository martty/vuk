#pragma once
#include <shared_mutex>
#include <unordered_map>
#include <plf_colony.h>
#include <vulkan/vulkan.hpp>

class Context;

namespace std {
	template <>
	struct hash<vk::GraphicsPipelineCreateInfo> {
		size_t operator()(vk::GraphicsPipelineCreateInfo const & x) const noexcept {
			return x.stageCount; // kek
		}
	};
};

namespace std {
	template <>
	struct hash<vk::RenderPassCreateInfo> {
		size_t operator()(vk::RenderPassCreateInfo const & x) const noexcept {
			return x.attachmentCount; // kek
		}
	};
};


namespace vuk {
	class Context;
	class InflightContext;

	template<class T>
	struct create_info;
	
	template<class T>
	using create_info_t = typename create_info<T>::type;

	template<class T>
	T create(Context& ctx, create_info_t<T> cinfo);

	template<> struct create_info<vk::Pipeline> {
		using type = vk::GraphicsPipelineCreateInfo;
	};

	template<> struct create_info<vk::RenderPass> {
		using type = vk::RenderPassCreateInfo;
	};


	template<class T>
	struct Cache {
		struct LRUEntry {
			T* ptr;
			unsigned last_use_frame;
		};

		Context& ctx;
		plf::colony<T> pool;
		std::unordered_map<create_info_t<T>, LRUEntry> lru_map; // possibly vector_map or an intrusive map
		std::shared_mutex cache_mtx;

		Cache(Context& ctx) : ctx(ctx) {}
		~Cache();

		struct View {
			InflightContext& ifc;
			Cache<T>& cache;

			View(InflightContext& ifc, Cache<T>& cache) : ifc(ifc), cache(cache) {}
			T acquire(create_info_t<T> ci);
			void collect(size_t threshold);
		};
	};
}
