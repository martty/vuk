#pragma once

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <vulkan/vulkan.hpp>
#include <unordered_set>
#include <algorithm>
#include <iostream>
#include <variant>
#include <string_view>
#include <optional>
#include <functional>
#include "Hash.hpp"

using Name = std::string_view;

struct Image {};

struct Attachment {
	Attachment(Name n) : name(n) {}

	Name name;
	enum class Type {
		eDepth, eColour, eInput
	} type;

	inline bool operator==(const Attachment& rhs) const {
		return name == rhs.name;
	}
};

struct Buffer {
	Name name;
	enum class Type {
		eStorage, eUniform, eTransfer
	} type;

	inline bool operator==(const Buffer& rhs) const {
		return name == rhs.name;
	}
};

namespace vuk {
	struct CommandBuffer;

	enum ImageAccess {
		eColorRW,
		eColorWrite,
		eColorRead,
		eDepthStencilRW,
		eDepthStencilRead,
		eInputRead,
		eVertexSampled,
		eVertexRead,
		eFragmentSampled,
		eFragmentRead,
		eFragmentWrite // written using image store
	};

	struct Resource;
	struct BufferResource {};
	struct ImageResource {
		Name name;

		Resource operator()(ImageAccess ia);
	};
}

inline vuk::ImageResource operator "" _image(const char* name, size_t) {
	return { name };
}

namespace vuk {
	struct Resource {
		Name name;
		enum class Type { eBuffer, eImage } type;
		ImageAccess ia;
		struct Use {
			vk::PipelineStageFlags stages;
			vk::AccessFlags access;
			vk::ImageLayout layout; // ignored for buffers
		};

		Resource(Name n, Type t, ImageAccess ia) : name(n), type(t), ia(ia) {}

		bool operator==(const Resource& o) const {
			return name == o.name;//std::tie(name, type) == std::tie(o.name, o.type);
		}
	};

	inline Resource ImageResource::operator()(ImageAccess ia) {
		return Resource{name, Resource::Type::eImage, ia};
	}

	struct Pass {
		Name name;
		Name executes_on;
		float auxiliary_order = 0.f;

		std::vector<Resource> resources;

		std::function<void(vuk::CommandBuffer&)> execute;
		//void(*execute)(struct CommandBuffer&);
	};
}

namespace std {
	template<> struct hash<vuk::Resource> {
		std::size_t operator()(vuk::Resource const& s) const noexcept {
			size_t h = 0;
			hash_combine(h, s.name, s.type);
			return h;
		}
	};
}


enum class BufferAccess {
	eWrite, eRead, eVertexAttributeRead
};
enum class ImageAccess {
	eTransferDst, eTransferSrc, eAttachmentWrite, eAttachmentInput, eShaderRead
};
enum class AccessType {
	eCompute, eGraphics, eHost
};

using QueueID = size_t;

struct BufferLifeCycle {
	QueueID last_queue;
	unsigned last_access;
	VkShaderStageFlagBits last_access_stage;
};

#include "Cache.hpp" // for create_info_t


template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...)->overloaded<Ts...>;
using io = std::variant<Buffer, Attachment>;

namespace std {
	template<> struct hash<Buffer> {
		std::size_t operator()(Buffer const& s) const noexcept {
			return std::hash<std::string_view>()(s.name);
		}
	};
}

namespace std {
	template<> struct hash<Attachment> {
		std::size_t operator()(Attachment const& s) const noexcept {
			return std::hash<std::string_view>()(s.name);
		}
	};
}

namespace std {
	template<> struct hash<io> {
		std::size_t operator()(io const& s) const noexcept {
			return std::visit([](const auto& v) -> size_t { return std::hash<std::remove_cvref_t<decltype(v)>>()(v); }, s);
		}
	};
}


inline bool operator==(const io& l, const io& r) {
	if (l.index() != r.index()) return false;
	return std::visit([&](const auto& v) {
		const auto& rhs = std::get<std::remove_cvref_t<decltype(v)>>(r);
		return v == rhs;
		}, l);
}

template<class T, class F>
T * contains_if(std::vector<T>& v, F&& f) {
	auto it = std::find_if(v.begin(), v.end(), f);
	if (it != v.end()) return &(*it);
	else return nullptr;
}

template<class T, class F>
T const * contains_if(const std::vector<T>& v, F&& f) {
	auto it = std::find_if(v.begin(), v.end(), f);
	if (it != v.end()) return &(*it);
	else return nullptr;
}

template<class T>
T const* contains(const std::vector<T>& v, const T& f) {
	auto it = std::find(v.begin(), v.end(), f);
	if (it != v.end()) return &(*it);
	else return nullptr;
}


inline std::string name_to_node(Name in) {
	std::string stripped = std::string(in);
	if (stripped.ends_with("[]")) stripped = stripped.substr(0, stripped.size() - 2);
	return stripped;
}

template <typename Iterator, typename Compare>
void topological_sort(Iterator begin, Iterator end, Compare cmp) {
	while (begin != end) {
		auto const new_begin = std::partition(begin, end, [&](auto const& a) {
			return std::none_of(begin, end, [&](auto const& b) { return cmp(b, a); });
			});
		assert(new_begin != begin && "not a partial ordering");
		begin = new_begin;
	}
}

#include <utility>

namespace vuk {
	struct RenderGraph {
		struct Sync {
			//std::vector<QueueXFer> transfers;
			/*std::vector<MemoryBarrier> membars;
			std::vector<ImageMemoryBarrier> imembars;*/
			struct QueueXfer {
				Name buffer;
				Name queue_src;
				Name queue_dst;
			};
			std::vector<QueueXfer> queue_transfers;
			std::vector<Name> signal_sema;
			std::vector<Name> wait_sema;
		};

		struct PassInfo {
			Pass pass;
			Sync sync_in;
			Sync sync_out;

			size_t render_pass_index;
			size_t subpass;

			std::unordered_set<Resource> inputs;
			std::unordered_set<Resource> outputs;

			std::unordered_set<Resource> global_inputs;
			std::unordered_set<Resource> global_outputs;

			bool is_head_pass = false;
			bool is_tail_pass = false;
		};

		std::vector<PassInfo> passes;

		std::vector<PassInfo*> head_passes;
		std::vector<PassInfo*> tail_passes;

		struct UseRef {
			Resource::Use use;
			PassInfo* pass = nullptr;
		};

		std::unordered_map<Name, std::vector<UseRef>> use_chains;

		struct AttachmentSInfo {
			vk::ImageLayout layout;
			vk::AccessFlags access;
			vk::PipelineStageFlags stage;
		};

		struct AttachmentRPInfo {
			Name name;
			vk::Extent2D extents;
			vuk::ImageView iv;
			vk::AttachmentDescription description;

			bool is_external = false;

			struct Use {
				vk::PipelineStageFlagBits stage;
				vk::AccessFlags access;
				size_t subpass;
			} first_use, last_use;
		};

		struct SubpassInfo {
			PassInfo* pass;
		};

		struct RenderPassInfo {
			std::vector<SubpassInfo> subpasses;
			std::vector<AttachmentRPInfo> attachments;
			vuk::RenderPassCreateInfo rpci;
			vuk::FramebufferCreateInfo fbci;
			vk::RenderPass handle;
			vk::Framebuffer framebuffer;
		};
		std::vector<RenderPassInfo> rpis;

		void add_pass(Pass p) {
			PassInfo pi;
			pi.pass = p;
			passes.push_back(pi);
		}

		std::unordered_set<Resource> global_inputs;
		std::unordered_set<Resource> global_outputs;
		std::vector<Resource> global_io;
		std::vector<Resource> tracked;

		// determine rendergraph inputs and outputs, and resources that are neither
		void build_io();

		void build();

		// RGscaffold
		std::unordered_map<Name, AttachmentRPInfo> bound_attachments;
		void bind_attachment_to_swapchain(Name name, vk::Format format, vk::Extent2D extent, vuk::ImageView siv);
		void mark_attachment_internal(Name, vk::Format, vk::Extent2D);

		// RG
		void build(vuk::PerThreadContext&);
		vk::CommandBuffer execute(vuk::PerThreadContext&);

		// debug
		void generate_graph_visualization();
	};
}
