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

#include <optional>
#include <functional>
struct Pass {
	Name name;
	Name executes_on;
	std::vector<Buffer> read_buffers; /* track read */
	std::vector<Buffer> write_buffers; /* track write */

	std::vector<Attachment> read_attachments;
	std::vector<Attachment> write_attachments;

	std::vector<Attachment> color_attachments;
	std::optional<Attachment> depth_attachment;

	std::function<void(struct CommandBuffer&)> execute;
	//void(*execute)(struct CommandBuffer&);
};

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

namespace vuk {
	class Context;
	class InflightContext;
}

#include "Cache.hpp"
struct CommandBuffer {
	Pass* current_pass;
	QueueID current_queue;
	Name bound_pipeline;
	vk::CommandBuffer command_buffer;
	vuk::InflightContext& ifc;

	CommandBuffer(vuk::InflightContext& ifc, vk::CommandBuffer cb) : ifc(ifc), command_buffer(cb) {}

	std::optional<std::pair<vk::RenderPass, uint32_t>> ongoing_renderpass;
	std::optional<vk::Viewport> next_viewport;
	std::optional<vk::Rect2D> next_scissor;
	std::optional<vuk::create_info_t<vk::Pipeline>> next_graphics_pipeline;

	// global memory barrier
	bool global_memory_barrier_inserted_since_last_draw = false;
	unsigned src_access_mask = 0;
	unsigned dst_access_mask = 0;
	// buffer barriers
	struct QueueXFer {
		QueueID from;
		QueueID to;
	};
	std::vector<QueueXFer> queue_transfers;
	
	void bind_pipeline(Name p) {
		bound_pipeline = p;
	}

	struct dynamic_state {

	};
	CommandBuffer& set_viewport(vk::Viewport vp) {
		next_viewport = vp;
		return *this;
	}

	CommandBuffer& set_scissor(vk::Rect2D vp) {
		next_scissor = vp;
		return *this;
	}

	CommandBuffer& bind_pipeline(vk::GraphicsPipelineCreateInfo gpci) {
		next_graphics_pipeline = gpci;
		return *this;
	}

	CommandBuffer& draw(uint32_t a, uint32_t b, uint32_t c, uint32_t d);
};

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

		std::unordered_set<io> inputs;
		std::unordered_set<io> outputs;

		std::unordered_set<io> global_inputs;
		std::unordered_set<io> global_outputs;

		bool is_head_pass = false;
		bool is_tail_pass = false;

		bool is_read_attachment(Name n) {
			return std::find_if(pass.read_attachments.begin(), pass.read_attachments.end(), [&](auto& att) { return att.name == n; }) != pass.read_attachments.end();
		}
	};

	std::vector<PassInfo> passes;

	std::vector<PassInfo*> head_passes;
	std::vector<PassInfo*> tail_passes;

	struct AttachmentSInfo {
		vk::ImageLayout layout;
	};

	struct AttachmentRPInfo {
		Name name;
		vk::Extent2D extents;
		vk::ImageView iv;
		vk::AttachmentDescription description;

		// sync in
		// the second half of sync_in is deduced
		vk::PipelineStageFlagBits srcStage;
		vk::AccessFlags srcAccess;
		// sync out
		// the first half of sync_out is deduced
		vk::PipelineStageFlagBits dstStage;
		vk::AccessFlags dstAccess;

		struct Use {
			vk::PipelineStageFlagBits stage;
			vk::AccessFlags access;
			size_t subpass;
		} first_use, last_use;
	};

	struct SubpassInfo {
		PassInfo* pass;
		std::unordered_map<Name, AttachmentSInfo> attachments;
	};
	struct RenderPassInfo {
		std::vector<SubpassInfo> subpasses;
		std::vector<AttachmentRPInfo> attachments;
		vk::RenderPassCreateInfo rpci;
		vk::FramebufferCreateInfo fbci;
		vk::RenderPass handle;
		vk::Framebuffer framebuffer;
	};
	std::vector<RenderPassInfo> rpis;

	void add_pass(Pass p) {
		PassInfo pi;
		pi.pass = p;
		passes.push_back(pi);
	}

	std::unordered_set<io> global_inputs;
	std::unordered_set<io> global_outputs;
	std::vector<io> global_io;
	std::vector<io> tracked;

	// determine rendergraph inputs and outputs, and resources that are neither
	void build_io() {
		std::unordered_set<io> inputs;
		std::unordered_set<io> outputs;

		for (auto& pif : passes) {
			pif.inputs.insert(pif.pass.read_attachments.begin(), pif.pass.read_attachments.end());
			pif.inputs.insert(pif.pass.color_attachments.begin(), pif.pass.color_attachments.end());
			if (pif.pass.depth_attachment) {
				pif.inputs.insert(*pif.pass.depth_attachment);
			}
			pif.outputs.insert(pif.pass.write_attachments.begin(), pif.pass.write_attachments.end());
			pif.outputs.insert(pif.pass.color_attachments.begin(), pif.pass.color_attachments.end());
			if (pif.pass.depth_attachment) {
				pif.outputs.insert(*pif.pass.depth_attachment);
			}
			
			for (auto& i : pif.inputs) {
				if (global_outputs.erase(i) == 0) {
					pif.global_inputs.insert(i);
				}
			}
			for (auto& i : pif.outputs) {
				if (global_inputs.erase(i) == 0) {
					pif.global_outputs.insert(i);
				}
			}

			global_inputs.insert(pif.global_inputs.begin(), pif.global_inputs.end());
			global_outputs.insert(pif.global_outputs.begin(), pif.global_outputs.end());

			inputs.insert(pif.inputs.begin(), pif.inputs.end());
			outputs.insert(pif.outputs.begin(), pif.outputs.end());
		}

		std::copy_if(outputs.begin(), outputs.end(), std::back_inserter(tracked), [&](auto& needle) { return !global_outputs.contains(needle); });
		global_io.insert(global_io.end(), global_inputs.begin(), global_inputs.end());
		global_io.insert(global_io.end(), global_outputs.begin(), global_outputs.end());
		global_io.erase(std::unique(global_io.begin(), global_io.end()), global_io.end());
	}

	void build() {
		// compute sync
		// find which reads are graph inputs (not produced by any pass) & outputs (not consumed by any pass)
		build_io();
		// sort passes
		if (passes.size() > 1) {
			topological_sort(passes.begin(), passes.end(), [](const auto& p1, const auto& p2) {
				for (auto& o : p1.outputs) {
					if (p2.inputs.contains(o)) return true;
				}
				return false;
				});
		}
		// determine which passes are "head" and "tail" -> ones that can execute in the beginning or the end of the RG
		// -> the ones that only have global io
		for (auto& p : passes) {
			if (p.global_inputs.size() == p.inputs.size()) {
				head_passes.push_back(&p);
				p.is_head_pass = true;
			}
			if (p.global_outputs.size() == p.outputs.size()) {
				tail_passes.push_back(&p);
				p.is_tail_pass = true;
			}
		}
		// go through all inputs and propagate dependencies onto last write pass
		for (auto& t : tracked) {
			std::visit(overloaded{
			[&](Buffer& th) {
					// for buffers, we need to track last write (can only be shader_write or transfer_write) + last write queue
					// if queues are different, we want to put a queue xfer on src and first dst + a semaphore signal on src and semaphore wait on first dst
					// if queues are the same, we want to put signalEvent on src and waitEvent on first dst OR pbarrier on first dst
					PassInfo* src = nullptr;
					PassInfo* dst = nullptr;
					Name write_queue;
					Name read_queue = "INVALID";
					vk::AccessFlags write_access;
					vk::AccessFlags read_access;
					vk::PipelineStageFlags write_stage;
					vk::PipelineStageFlags read_stage;
					for (auto& p : passes) {
						if (contains(p.pass.write_buffers, th)) {
							src = &p;
							write_queue = p.pass.executes_on;
							write_access = (th.type == Buffer::Type::eStorage || th.type == Buffer::Type::eUniform) ?
											vk::AccessFlagBits::eShaderWrite : vk::AccessFlagBits::eTransferWrite;
							write_stage = (th.type == Buffer::Type::eStorage || th.type == Buffer::Type::eUniform) ?
										 vk::PipelineStageFlagBits::eAllGraphics : vk::PipelineStageFlagBits::eTransfer;
						}

						if (contains(p.pass.read_buffers, th)) {
							if (!dst)
								dst = &p;
							// handle a single type of dst queue for now
							assert(read_queue == "INVALID" || read_queue == p.pass.executes_on);
							read_queue = p.pass.executes_on;
							read_access |= (th.type == Buffer::Type::eStorage || th.type == Buffer::Type::eUniform) ?
											vk::AccessFlagBits::eShaderRead : vk::AccessFlagBits::eTransferRead;
							read_stage |= (th.type == Buffer::Type::eStorage || th.type == Buffer::Type::eUniform) ?
										 vk::PipelineStageFlagBits::eAllGraphics : vk::PipelineStageFlagBits::eTransfer;
						}
					}
					bool queue_xfer = write_queue != read_queue;
					assert(src);
					auto& sync_out = src->sync_out;
					assert(dst);
					auto& sync_in = dst->sync_in;
					if (queue_xfer) {
						Sync::QueueXfer xfer;
						xfer.queue_src = write_queue;
						xfer.queue_dst = read_queue;
						xfer.buffer = th.name;
						sync_out.queue_transfers.push_back(xfer);
						sync_in.queue_transfers.push_back(xfer);
						sync_out.signal_sema.push_back(src->pass.name);
						sync_in.wait_sema.push_back(src->pass.name);
					}
				},
				[&](Attachment& th) {
				for (auto& p : passes) {
				}
				} }, t);
		}

		// we need to collect passes into framebuffers, which will determine the renderpasses
		std::vector<std::pair<std::unordered_set<Attachment>, std::vector<PassInfo*>>> attachment_sets;
		for (auto& passinfo : passes) {
			std::unordered_set<Attachment> atts;
			atts.insert(passinfo.pass.color_attachments.begin(), passinfo.pass.color_attachments.end());
			if(passinfo.pass.depth_attachment) atts.insert(*passinfo.pass.depth_attachment);
			
			if (auto p = contains_if(attachment_sets, [&](auto& t) { return t.first == atts; })) {
				p->second.push_back(&passinfo);
			} else {
				attachment_sets.emplace_back(atts, std::vector{&passinfo});
			}
		}

		// renderpasses are uniquely identified by their index from now on
		// tell passes in which renderpass/subpass they will execute
		for (auto& set : attachment_sets) {
			RenderPassInfo rpi;
			auto rpi_index = rpis.size();

			size_t subpass = 0;
			for (auto& p : set.second) {
				p->render_pass_index = rpi_index;
				p->subpass = subpass++;
				SubpassInfo si;
				si.pass = p;
				for (auto& a : p->pass.color_attachments) {
					si.attachments.emplace(a.name, AttachmentSInfo{vk::ImageLayout::eColorAttachmentOptimal});
					// TODO: ColorAttachmentRead happens if blending or logicOp
					if (contains_if(rpi.attachments, [&](auto& att) { return att.name == a.name; })) {
						// TODO: we want to add all same level passes here, otherwise the other passes might not get the right sync
					} else {
						AttachmentRPInfo arpi;
						arpi.name = a.name;
						arpi.first_use.access = vk::AccessFlagBits::eColorAttachmentWrite;
						arpi.first_use.stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
						arpi.first_use.subpass = subpass - 1;

						rpi.attachments.push_back(arpi);
					}
				}
				rpi.subpasses.push_back(si);
			}
			rpis.push_back(rpi);
		}

	}

	std::unordered_map<Name, AttachmentRPInfo> bound_attachments;

	void bind_attachment_to_swapchain(Name name, vk::Format format, vk::Extent2D extent, vk::ImageView siv) {
		AttachmentRPInfo attachment_info;
		attachment_info.extents = extent;
		attachment_info.iv = siv;
		// for WSI attachments we don't want to preserve previous data
		attachment_info.description.initialLayout = vk::ImageLayout::eUndefined;
		// directly presented
		attachment_info.description.finalLayout = vk::ImageLayout::ePresentSrcKHR;
		attachment_info.description.loadOp = vk::AttachmentLoadOp::eClear;
		attachment_info.description.storeOp = vk::AttachmentStoreOp::eStore;

		attachment_info.description.format = format;
		attachment_info.description.samples = vk::SampleCountFlagBits::e1;

		// for WSI, we want to wait for colourattachmentoutput
		attachment_info.srcStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		// we don't care about any writes, we will clear
		attachment_info.srcAccess = vk::AccessFlags{};
		bound_attachments.emplace(name, attachment_info);
	}

	void build(vuk::InflightContext&);

	vk::CommandBuffer execute(vuk::InflightContext&);

	void generate_graph_visualization();
};
