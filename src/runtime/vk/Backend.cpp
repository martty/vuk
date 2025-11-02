#include "vuk/Hash.hpp" // for create
#include "vuk/ir/IRPass.hpp"
#include "vuk/ir/IRProcess.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/runtime/CommandBuffer.hpp"
#include "vuk/runtime/Stream.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/runtime/vk/RenderPass.hpp"
#include "vuk/runtime/vk/VkQueueExecutor.hpp"
#include "vuk/runtime/vk/VkRuntime.hpp"
#include "vuk/SyncLowering.hpp"

#include <fmt/format.h>
#include <unordered_set>
#include <vector>

// #define VUK_DUMP_EXEC
// #define VUK_DEBUG_IMBAR
// #define VUK_DEBUG_MEMBAR

namespace vuk {
	struct RenderPassInfo {
		std::vector<VkImageView> framebuffer_ivs;
		RenderPassCreateInfo rpci;
		FramebufferCreateInfo fbci;
		VkRenderPass handle = {};
		VkFramebuffer framebuffer;
	};

	void begin_render_pass(Runtime& ctx, RenderPassInfo& rpass, VkCommandBuffer& cbuf, bool use_secondary_command_buffers) {
		VkRenderPassBeginInfo rbi{ .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
		rbi.renderPass = rpass.handle;
		rbi.framebuffer = rpass.framebuffer;
		rbi.renderArea = VkRect2D{ Offset2D{}, Extent2D{ rpass.fbci.width, rpass.fbci.height } };
		rbi.clearValueCount = 0;

		ctx.vkCmdBeginRenderPass(cbuf, &rbi, use_secondary_command_buffers ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
	}

	struct VkQueueStream : public Stream {
		Runtime& ctx;
		QueueExecutor* executor;

		std::vector<SubmitInfo> batch;
		std::deque<Signal> signals;
		SubmitInfo si;
		Unique<CommandPool> cpool;
		Unique<CommandBufferAllocation> hl_cbuf;
		VkCommandBuffer cbuf = VK_NULL_HANDLE;
		ProfilingCallbacks* callbacks;
		bool is_recording = false;
		void* cbuf_profile_data = nullptr;

		RenderPassInfo rp = {};
		std::vector<VkImageMemoryBarrier2KHR> im_bars;
		std::vector<VkImageMemoryBarrier2KHR> half_im_bars;
		std::vector<VkMemoryBarrier2KHR> mem_bars;
		std::vector<VkMemoryBarrier2KHR> half_mem_bars;

		VkQueueStream(Allocator alloc, QueueExecutor* qe, ProfilingCallbacks* callbacks) :
		    Stream(alloc, qe),
		    ctx(alloc.get_context()),
		    executor(qe),
		    callbacks(callbacks) {
			domain = qe->tag.domain;
		}

		void add_dependency(Stream* dep) override {
			dependencies.push_back(dep);
			if (dep->domain == DomainFlagBits::eHost) {
				return;
			}
			if (is_recording) {
				end_cbuf();
				batch.emplace_back();
			}
		}

		Signal* make_signal() override {
			return &signals.emplace_back();
		}

		void sync_deps() override {
			if (batch.empty()) {
				batch.emplace_back();
			}
			for (auto dep : dependencies) {
				auto signal = dep->make_signal();
				if (signal) {
					dep->add_dependent_signal(signal);
				}
				auto res = *dep->submit();
				if (signal) {
					batch.back().waits.push_back(signal);
				}
				if (res.sema_wait != VK_NULL_HANDLE) {
					batch.back().pres_wait.push_back(res.sema_wait);
				}
			}
			dependencies.clear();
			if (!is_recording) {
				begin_cbuf();
			}
			flush_barriers();
		}

		Result<SubmitResult> submit() override {
			sync_deps();
			end_cbuf();
			for (auto& signal : dependent_signals) {
				signal->source.executor = executor;
				batch.back().signals.emplace_back(signal);
			}
			executor->submit_batch(batch);
			for (auto& item : batch) {
				for (auto& signal : item.signals) {
					alloc.wait_sync_points(std::span{ &signal->source, 1 });
				}
			}
			batch.clear();

			// propagate signal to nodes in submit scope
			auto& propsig = dependent_signals.back();
			for (auto& node : current_submit) {
				if (node->rel_acq) {
					node->rel_acq->status = propsig->status;
					node->rel_acq->source = propsig->source;
				}
			}
			current_submit.clear();
			dependent_signals.clear();
			return { expected_value };
		}

		Result<VkResult> present(Swapchain& swp) {
			batch.back().pres_signal.emplace_back(swp.semaphores[swp.image_index]);
			submit();
			VkPresentInfoKHR pi{ .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
			pi.swapchainCount = 1;
			pi.pSwapchains = &swp.swapchain;
			pi.pImageIndices = &swp.image_index;
			pi.waitSemaphoreCount = 1;
			pi.pWaitSemaphores = &swp.semaphores[swp.image_index];
			auto res = executor->queue_present(pi);
			if (res && swp.acquire_result == VK_SUBOPTIMAL_KHR) {
				return { expected_value, VK_SUBOPTIMAL_KHR };
			}
			return res;
		}

		Result<void> begin_cbuf() {
			assert(!is_recording);
			is_recording = true;
			domain = domain;
			if (cpool->command_pool == VK_NULL_HANDLE) {
				cpool = Unique<CommandPool>(alloc);
				VkCommandPoolCreateInfo cpci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
				cpci.flags = VkCommandPoolCreateFlagBits::VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
				cpci.queueFamilyIndex = executor->get_queue_family_index(); // currently queue family idx = queue idx

				VUK_DO_OR_RETURN(alloc.allocate_command_pools(std::span{ &*cpool, 1 }, std::span{ &cpci, 1 }));
			}
			hl_cbuf = Unique<CommandBufferAllocation>(alloc);
			CommandBufferAllocationCreateInfo ci{ .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .command_pool = *cpool };
			VUK_DO_OR_RETURN(alloc.allocate_command_buffers(std::span{ &*hl_cbuf, 1 }, std::span{ &ci, 1 }));

			si.command_buffers.emplace_back(*hl_cbuf);

			cbuf = hl_cbuf->command_buffer;

			VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
			alloc.get_context().vkBeginCommandBuffer(cbuf, &cbi);

			cbuf_profile_data = nullptr;
			if (callbacks->on_begin_command_buffer) {
				cbuf_profile_data = callbacks->on_begin_command_buffer(callbacks->user_data, executor->tag, cbuf);
			}

			return { expected_value };
		}

		Result<void> end_cbuf() {
			flush_barriers();
			is_recording = false;
			if (callbacks->on_end_command_buffer)
				callbacks->on_end_command_buffer(callbacks->user_data, cbuf_profile_data);
			if (auto result = ctx.vkEndCommandBuffer(hl_cbuf->command_buffer); result != VK_SUCCESS) {
				return { expected_error, VkException{ result } };
			}
			batch.back().command_buffers.push_back(hl_cbuf->command_buffer);
			cbuf = VK_NULL_HANDLE;
			return { expected_value };
		};

		void flush_barriers() {
			VkDependencyInfoKHR dependency_info{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO_KHR,
				                                   .memoryBarrierCount = (uint32_t)mem_bars.size(),
				                                   .pMemoryBarriers = mem_bars.data(),
				                                   .imageMemoryBarrierCount = (uint32_t)im_bars.size(),
				                                   .pImageMemoryBarriers = im_bars.data() };

			if (mem_bars.size() > 0 || im_bars.size() > 0) {
				ctx.vkCmdPipelineBarrier2KHR(cbuf, &dependency_info);
			}

			mem_bars.clear();
			im_bars.clear();
		}

		void print_ib(VkImageMemoryBarrier2KHR ib, std::string extra = "") {
			auto layout_to_str = [](VkImageLayout l) {
				switch (l) {
				case VK_IMAGE_LAYOUT_UNDEFINED:
					return "UND";
				case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
					return "SRC";
				case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
					return "DST";
				case VK_IMAGE_LAYOUT_GENERAL:
					return "GEN";
				case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL:
					return "ROO";
				case VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL:
					return "ATT";
				case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
					return "PRS";
				default:
					assert(0);
					return "";
				}
			};
			fmt::println("[{}][m{}:{}][l{}:{}][{}->{}]{}",
			             fmt::ptr(ib.image),
			             ib.subresourceRange.baseMipLevel,
			             ib.subresourceRange.baseMipLevel + ib.subresourceRange.levelCount - 1,
			             ib.subresourceRange.baseArrayLayer,
			             ib.subresourceRange.baseArrayLayer + ib.subresourceRange.layerCount - 1,
			             layout_to_str(ib.oldLayout),
			             layout_to_str(ib.newLayout),
			             extra);
		}

		bool is_readonly_layout(VkImageLayout l) {
			switch (l) {
			case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
			case VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL:
				return true;
			default:
				return false;
			}
		}

		void synch_image(ImageAttachment& img_att, Subrange::Image subrange, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override {
			auto aspect = format_to_aspect(img_att.format);

			// if we start an RP and we have LOAD_OP_LOAD (currently always), then we must upgrade access with an appropriate READ
			if (is_framebuffer_attachment(dst_use)) {
				if ((aspect & ImageAspectFlagBits::eColor) == ImageAspectFlags{}) { // not color -> depth or depth/stencil
					dst_use.access |= AccessFlagBits::eDepthStencilAttachmentRead;
				} else {
					dst_use.access |= AccessFlagBits::eColorAttachmentRead;
				}
			}

			DomainFlagBits src_domain = src_use.stream ? src_use.stream->domain : DomainFlagBits::eNone;
			DomainFlagBits dst_domain = dst_use.stream ? dst_use.stream->domain : DomainFlagBits::eNone;

			scope_to_domain((VkPipelineStageFlagBits2KHR&)src_use.stages, src_domain & DomainFlagBits::eQueueMask);
			scope_to_domain((VkPipelineStageFlagBits2KHR&)dst_use.stages, dst_domain & DomainFlagBits::eQueueMask);

			// compute image barrier for this access -> access
			VkImageMemoryBarrier2KHR barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2_KHR };
			barrier.srcAccessMask = is_readonly_access(src_use) ? 0 : (VkAccessFlags2)src_use.access;
			barrier.dstAccessMask = (VkAccessFlags2)dst_use.access;
			barrier.oldLayout = (VkImageLayout)src_use.layout;
			barrier.newLayout = (VkImageLayout)dst_use.layout;
			barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
			barrier.subresourceRange.baseArrayLayer = subrange.base_layer;
			barrier.subresourceRange.baseMipLevel = subrange.base_level;
			barrier.subresourceRange.layerCount = subrange.layer_count;
			barrier.subresourceRange.levelCount = subrange.level_count;

			if (src_domain == DomainFlagBits::eAny || src_domain == DomainFlagBits::eHost) {
				src_domain = dst_domain;
			}
			if (dst_domain == DomainFlagBits::eAny) {
				dst_domain = src_domain;
			}

			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			if (src_use.stream && dst_use.stream && src_use.stream != dst_use.stream) { // cross-stream
				if (src_use.stream->executor && src_use.stream->executor->type == Executor::Type::eVulkanDeviceQueue && dst_use.stream->executor &&
				    dst_use.stream->executor->type == Executor::Type::eVulkanDeviceQueue) { // cross queue
					auto src_queue = static_cast<QueueExecutor*>(src_use.stream->executor);
					auto dst_queue = static_cast<QueueExecutor*>(dst_use.stream->executor);
					if (src_queue->get_queue_family_index() != dst_queue->get_queue_family_index()) { // cross queue family
						barrier.srcQueueFamilyIndex = src_queue->get_queue_family_index();
						barrier.dstQueueFamilyIndex = dst_queue->get_queue_family_index();
					}
				}
			}

			if (src_use.stages == PipelineStageFlags{}) {
				barrier.srcAccessMask = {};
			}
			if (dst_use.stages == PipelineStageFlags{}) {
				barrier.dstAccessMask = {};
			}

			barrier.srcStageMask = (VkPipelineStageFlags2)src_use.stages.m_mask;
			barrier.dstStageMask = (VkPipelineStageFlags2)dst_use.stages.m_mask;

			barrier.image = img_att.image.image;

#ifdef VUK_DEBUG_IMBAR
			print_ib(barrier, "$");
#endif
			// assert(img_att.layout == ImageLayout::eUndefined || barrier.oldLayout != VK_IMAGE_LAYOUT_UNDEFINED);
			assert(barrier.oldLayout != VK_IMAGE_LAYOUT_UNDEFINED || !is_readonly_layout(barrier.newLayout));
			im_bars.push_back(barrier);

			img_att.layout = (ImageLayout)barrier.newLayout;
			if (barrier.oldLayout != VK_IMAGE_LAYOUT_UNDEFINED) {
				assert(barrier.newLayout != VK_IMAGE_LAYOUT_UNDEFINED);
			}
		};

		void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override {
			VkMemoryBarrier2KHR barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2_KHR };

			DomainFlagBits src_domain = src_use.stream ? src_use.stream->domain : DomainFlagBits::eNone;
			DomainFlagBits dst_domain = dst_use.stream ? dst_use.stream->domain : DomainFlagBits::eNone;

			if (src_domain == DomainFlagBits::eAny || dst_domain == DomainFlagBits::eHost) {
				src_domain = dst_domain;
			}
			if (dst_domain == DomainFlagBits::eAny) {
				dst_domain = src_domain;
			}

			// always dst domain - we don't emit "release" on the src stream
			scope_to_domain((VkPipelineStageFlagBits2KHR&)src_use.stages, dst_domain & DomainFlagBits::eQueueMask);
			scope_to_domain((VkPipelineStageFlagBits2KHR&)dst_use.stages, dst_domain & DomainFlagBits::eQueueMask);

			barrier.srcAccessMask = is_readonly_access(src_use) ? 0 : (VkAccessFlags2)src_use.access;
			barrier.dstAccessMask = (VkAccessFlags2)dst_use.access;
			barrier.srcStageMask = (VkPipelineStageFlagBits2)src_use.stages.m_mask;
			barrier.dstStageMask = (VkPipelineStageFlagBits2)dst_use.stages.m_mask;
			if (barrier.srcStageMask == 0) {
				barrier.srcStageMask = (VkPipelineStageFlagBits2)PipelineStageFlagBits::eNone;
				barrier.srcAccessMask = {};
			}

			mem_bars.push_back(barrier);
		};

		void prepare_render_pass_attachment(Allocator alloc, ImageAttachment img_att) {
			auto aspect = format_to_aspect(img_att.format);
			VkAttachmentReference attref{};

			attref.attachment = (uint32_t)rp.rpci.attachments.size();

			auto& descr = rp.rpci.attachments.emplace_back();
			// no layout changed by RPs currently
			descr.initialLayout = (VkImageLayout)img_att.layout;
			descr.finalLayout = (VkImageLayout)img_att.layout;
			attref.layout = (VkImageLayout)img_att.layout;

			descr.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
			descr.storeOp = is_readonly_layout((VkImageLayout)img_att.layout) ? VK_ATTACHMENT_STORE_OP_NONE_KHR : VK_ATTACHMENT_STORE_OP_STORE;

			descr.format = (VkFormat)img_att.format;
			descr.samples = (VkSampleCountFlagBits)img_att.sample_count.count;

			if ((aspect & ImageAspectFlagBits::eColor) == ImageAspectFlags{}) { // not color -> depth or depth/stencil
				rp.rpci.ds_ref = attref;
			} else {
				rp.rpci.color_refs.push_back(attref);
			}

			if (img_att.image_view.payload == VK_NULL_HANDLE) {
				auto iv = allocate_image_view(alloc, img_att); // TODO: dropping error
				img_att.image_view = **iv;

				alloc.get_context().set_name(img_att.image_view.payload, Name("ImageView: RenderTarget "));
			}
			rp.framebuffer_ivs.push_back(img_att.image_view.payload);
			rp.fbci.width = img_att.extent.width;
			rp.fbci.height = img_att.extent.height;
			rp.fbci.layers = img_att.layer_count;
			assert(img_att.level_count == 1);
			rp.fbci.sample_count = img_att.sample_count;
			rp.fbci.attachments.push_back(img_att.image_view);
		}

		Result<void> prepare_render_pass() {
			SubpassDescription sd;
			sd.colorAttachmentCount = (uint32_t)rp.rpci.color_refs.size();
			sd.pColorAttachments = rp.rpci.color_refs.data();

			sd.pDepthStencilAttachment = rp.rpci.ds_ref ? &*rp.rpci.ds_ref : nullptr;
			sd.flags = {};
			sd.inputAttachmentCount = 0;
			sd.pInputAttachments = nullptr;
			sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			sd.preserveAttachmentCount = 0;
			sd.pPreserveAttachments = nullptr;

			rp.rpci.subpass_descriptions.push_back(sd);

			rp.rpci.subpassCount = (uint32_t)rp.rpci.subpass_descriptions.size();
			rp.rpci.pSubpasses = rp.rpci.subpass_descriptions.data();

			// we use barriers
			rp.rpci.dependencyCount = 0;
			rp.rpci.pDependencies = nullptr;

			rp.rpci.attachmentCount = (uint32_t)rp.rpci.attachments.size();
			rp.rpci.pAttachments = rp.rpci.attachments.data();

			auto result = alloc.allocate_render_passes(std::span{ &rp.handle, 1 }, std::span{ &rp.rpci, 1 });

			rp.fbci.renderPass = rp.handle;
			rp.fbci.pAttachments = rp.framebuffer_ivs.data();
			rp.fbci.attachmentCount = (uint32_t)rp.framebuffer_ivs.size();

			Unique<VkFramebuffer> fb(alloc);
			VUK_DO_OR_RETURN(alloc.allocate_framebuffers(std::span{ &*fb, 1 }, std::span{ &rp.fbci, 1 }));
			rp.framebuffer = *fb; // queue framebuffer for destruction
			// drop render pass immediately
			if (result) {
				alloc.deallocate(std::span{ &rp.handle, 1 });
			}
			begin_render_pass(alloc.get_context(), rp, cbuf, false);

			return { expected_value };
		}

		void end_render_pass() {
			alloc.get_context().vkCmdEndRenderPass(cbuf);
			rp = {};
		}
	};

	struct HostStream : Stream {
		HostStream(Allocator alloc) : Stream(alloc, nullptr) {
			domain = DomainFlagBits::eHost;
		}

		void add_dependent_signal(Signal* signal) override {
			signal->source.executor = executor;
			signal->source.visibility = 0;
			signal->status = Signal::Status::eHostAvailable;
		}

		void add_dependency(Stream* dep) override {
			dependencies.push_back(dep);
		}
		void sync_deps() override {
			assert(false);
		}

		void synch_image(ImageAttachment& img_att, Subrange::Image subrange, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override {
			/* host -> host and host -> device not needed, device -> host inserts things on the device side */
			return;
		}
		void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override {
			/* host -> host and host -> device not needed, device -> host inserts things on the device side */
			return;
		}

		Signal* make_signal() override {
			return nullptr;
		}

		Result<SubmitResult> submit() override {
			for (auto& sig : dependent_signals) {
				sig->status = Signal::Status::eHostAvailable;
			}
			for (auto& node : current_submit) {
				if (node->rel_acq) {
					node->rel_acq->status = Signal::Status::eHostAvailable;
					node->rel_acq->source.executor = executor;
					node->rel_acq->source.visibility = 0;
				}
			}
			return { expected_value };
		}
	};

	struct VkPEStream : Stream {
		VkPEStream(Allocator alloc, Swapchain& swp, VkSemaphore acquire_sema) : Stream(alloc, nullptr), swp(&swp), acquire_sema(acquire_sema) {
			domain = DomainFlagBits::ePE;
		}

		void add_dependency(Stream* dep) override {
			dependencies.push_back(dep);
		}
		void sync_deps() override {
			assert(false);
		}

		void synch_image(ImageAttachment& img_att, Subrange::Image subrange, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override {}

		void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) override { /* PE doesn't do memory */
			assert(false);
		}

		Signal* make_signal() override {
			return nullptr;
		}

		Result<SubmitResult> submit() override {
			assert(swp);
			SubmitResult sr{ .sema_wait = acquire_sema };
			return { expected_value, sr };
		}

		Swapchain* swp;
		VkSemaphore acquire_sema;
	};

	struct Recorder {
		Recorder(Allocator alloc, ProfilingCallbacks* callbacks, std::pmr::vector<Ref>& pass_reads) :
		    ctx(alloc.get_context()),
		    alloc(alloc),
		    callbacks(callbacks),
		    pass_reads(pass_reads) {
			last_modify.emplace(0, new (this->arena.ensure_space(sizeof(PartialStreamResourceUse))) PartialStreamResourceUse{ { to_use(eNone), nullptr } });
		}
		Runtime& ctx;
		Allocator alloc;
		ProfilingCallbacks* callbacks;
		std::pmr::vector<Ref>& pass_reads;
		InlineArena<std::byte, 1024> arena;

		std::unordered_map<DomainFlagBits, std::unique_ptr<Stream>> streams;
		struct PartialStreamResourceUse : StreamResourceUse {
			Subrange subrange;
			PartialStreamResourceUse* prev = nullptr;
			PartialStreamResourceUse* next = nullptr;
		};

		std::unordered_map<uint64_t, PartialStreamResourceUse*> last_modify;

		// start recording if needed
		// all dependant domains flushed
		// all pending sync to be flushed
		void synchronize_stream(Stream* stream) {
			stream->sync_deps();
		}

		Stream* stream_for_domain(DomainFlagBits domain) {
			for (auto& [dom, stream] : streams) {
				if (dom & domain) {
					return stream.get();
				}
			}
			return nullptr;
		}

		Stream* stream_for_executor(Executor* executor) {
			for (auto& [domain, stream] : streams) {
				if (stream->executor == executor) {
					return stream.get();
				}
			}
			assert(0);
			return nullptr;
		}

		uint64_t value_identity(Type* base_ty, void* value) {
			uint64_t key = 0;
			if (base_ty->hash_value == current_module->types.builtin_image) {
				auto& img_att = *reinterpret_cast<ImageAttachment*>(value);
				key = reinterpret_cast<uint64_t>(img_att.image.image);
			} else if (base_ty->is_bufferlike_view()) {
				auto buf = reinterpret_cast<Buffer<>*>(value);
				auto bo = alloc.get_context().ptr_to_buffer_offset(buf->ptr);
				key = reinterpret_cast<uint64_t>(bo.buffer);
			} else if (base_ty->kind == Type::ARRAY_TY) {
				if (base_ty->array.count > 0) { // for an array, we key off the the first element, as the array syncs together
					auto elem_ty = base_ty->array.T->get();
					auto elems = reinterpret_cast<std::byte*>(value);
					return value_identity(elem_ty, elems);
				} else { // zero-len arrays
					return 0;
				}
			} else if (base_ty->hash_value == current_module->types.builtin_sampled_image) { // only image syncs
				auto& img_att = reinterpret_cast<SampledImage*>(value)->ia;
				key = reinterpret_cast<uint64_t>(img_att.image.image);
			} else {
				return 0;
			}
			return key;
		}

		void init_sync(Type* base_ty, StreamResourceUse src_use, void* value, bool enforce_unique = true) {
			if (base_ty->kind == Type::ARRAY_TY) { // for an array, we init all elements
				auto elem_ty = base_ty->array.T->get();
				auto size = base_ty->array.count;
				auto elems = reinterpret_cast<std::byte*>(value);
				for (int i = 0; i < size; i++) {
					init_sync(elem_ty, src_use, elems, enforce_unique);
					elems += elem_ty->size;
				}
				return;
			} else if (base_ty->kind == Type::COMPOSITE_TY) { // do each member for a composite
				if (!base_ty->is_bufferlike_view()) {           // if the type is a view, we will sync it, otherwise sync each elem
					for (size_t i = 0; i < base_ty->composite.types.size(); i++) {
						init_sync(base_ty->composite.types[i].get(), src_use, base_ty->composite.get(value, i), enforce_unique);
					}
					return;
				}
			}

			uint64_t key = value_identity(base_ty, value);
			auto& psru = *new (this->arena.ensure_space(sizeof(PartialStreamResourceUse))) PartialStreamResourceUse{ src_use };
			if (base_ty->hash_value == current_module->types.builtin_image) {
				auto& img_att = *reinterpret_cast<ImageAttachment*>(value);
				psru.subrange.image = { img_att.base_level, img_att.level_count, img_att.base_layer, img_att.layer_count };
			} else if (base_ty->is_bufferlike_view()) { // for buffers, we allow underlying resource to alias
				auto buf = reinterpret_cast<Buffer<>*>(value);
				auto bo = alloc.get_context().ptr_to_buffer_offset(buf->ptr);
				// TODO: here we need to get the offset into the VkBuffer
				psru.subrange.buffer = { bo.offset, buf->sz_bytes };

				auto [v, succ] = last_modify.try_emplace(key, &psru);
				if (!succ) {
					auto head = v->second;
					while (head->next) {
						head = head->next;
					}
					head->next = &psru;
					psru.prev = head;
				}
				return;
			}

			if (enforce_unique && key != 0) {
				assert(last_modify.find(key) == last_modify.end());
				last_modify.emplace(key, &psru);
			} else {
				last_modify.try_emplace(key, &psru);
			}
		}

		void add_sync(Type* base_ty, std::optional<StreamResourceUse> maybe_dst_use, void* value) {
			if (!maybe_dst_use) {
				return;
			}
			auto& dst_use = *maybe_dst_use;

			if (base_ty->kind == Type::ARRAY_TY) {
				auto elem_ty = base_ty->array.T->get();
				auto size = base_ty->array.count;
				auto elems = reinterpret_cast<std::byte*>(value);
				for (int i = 0; i < size; i++) {
					add_sync(elem_ty, dst_use, elems);
					elems += elem_ty->size;
				}
				return;
			} else if (base_ty->hash_value == current_module->types.builtin_sampled_image) { // sync the image
				auto& img_att = reinterpret_cast<SampledImage*>(value)->ia;
				add_sync(current_module->types.get_builtin_image().get(), dst_use, &img_att);
				return;
			} else if (!base_ty->is_bufferlike_view() && base_ty->kind == Type::COMPOSITE_TY) { // sync every part of a composite
				auto& composite = base_ty->composite;
				for (int i = 0; i < composite.types.size(); i++) {
					add_sync(composite.types[i].get(), dst_use, base_ty->composite.get(value, i));
				}
				return;
			}

			uint64_t key = value_identity(base_ty, value);

			if (key == 0) { // doesn't require sync
				return;
			}

			auto& head = last_modify.at(key);

			if (base_ty->hash_value == current_module->types.builtin_image) {
				auto& img_att = *reinterpret_cast<ImageAttachment*>(value);
				std::vector<Subrange::Image, inline_alloc<Subrange::Image, 1024>> work_queue(this->arena);
				work_queue.emplace_back(Subrange::Image{ img_att.base_level, img_att.level_count, img_att.base_layer, img_att.layer_count });

				while (work_queue.size() > 0) {
					Subrange::Image dst_range = work_queue.back();
					Subrange::Image src_range, isection;
					work_queue.pop_back();
					auto src = head;
					assert(src);
					for (; src != nullptr; src = src->next) {
						src_range = { src->subrange.image.base_level, src->subrange.image.level_count, src->subrange.image.base_layer, src->subrange.image.layer_count };

						// we want to make a barrier for the intersection of the source and incoming
						auto isection_opt = intersect_one(src_range, dst_range);
						if (isection_opt) {
							isection = *isection_opt;
							break;
						}
					}
					assert(src);
					// remove the existing barrier from the candidates
					auto found = src;

					// wind to the end
					for (; src->next != nullptr; src = src->next)
						;
					// splinter the source and destination ranges
					difference_one(src_range, isection, [&](Subrange::Image nb) {
						// push the splintered src uses
						PartialStreamResourceUse psru{ *src };
						psru.subrange.image = { nb.base_level, nb.level_count, nb.base_layer, nb.layer_count };
						src->next = new (this->arena.ensure_space(sizeof(PartialStreamResourceUse))) PartialStreamResourceUse(psru);
						src->next->prev = src;
						src = src->next;
					});

					// splinter the dst uses, and push into the work queue
					difference_one(dst_range, isection, [&](Subrange::Image nb) { work_queue.push_back(nb); });

					auto& src_use = *found;
					if (src_use.stream && dst_use.stream && (src_use.stream != dst_use.stream)) {
						dst_use.stream->add_dependency(src_use.stream);
					}
					if (src_use.stream != dst_use.stream) {
						src_use.stream->synch_image(img_att, isection, src_use, dst_use, value); // synchronize dst onto first stream
					}
					dst_use.stream->synch_image(img_att, isection, src_use, dst_use, value); // synchronize src onto second stream

					static_cast<StreamResourceUse&>(*found) = dst_use;
					found->subrange.image.base_level = isection.base_level;
					found->subrange.image.level_count = isection.level_count;
					found->subrange.image.base_layer = isection.base_layer;
					found->subrange.image.layer_count = isection.layer_count;
				}
			} else if (base_ty->is_bufferlike_view()) {
				auto& att = *reinterpret_cast<Buffer<>*>(value);
				if (att.size == 0) {
					return;
				}
				auto bo = alloc.get_context().ptr_to_buffer_offset(att.ptr);
				std::vector<Subrange::Buffer, inline_alloc<Subrange::Buffer, 1024>> work_queue(this->arena);
				work_queue.emplace_back(Subrange::Buffer{ bo.offset, att.sz_bytes });

				while (work_queue.size() > 0) {
					Subrange::Buffer dst_range = work_queue.back();
					Subrange::Buffer src_range, isection;
					work_queue.pop_back();
					auto src = head;
					assert(src);
					for (; src != nullptr; src = src->next) {
						src_range = { src->subrange.buffer.offset, src->subrange.buffer.size };

						// we want to make a barrier for the intersection of the source and incoming
						auto isection_opt = intersect_one(src_range, dst_range);
						if (isection_opt) {
							isection = *isection_opt;
							break;
						}
					}
					assert(src);
					// remove the existing barrier from the candidates
					auto found = src;

					// wind to the end
					for (; src->next != nullptr; src = src->next)
						;
					// splinter the source and destination ranges
					difference_one(src_range, isection, [&](Subrange::Buffer nb) {
						// push the splintered src uses
						PartialStreamResourceUse psru{ *src };
						psru.subrange.buffer = { nb.offset, nb.size };
						src->next = new (this->arena.ensure_space(sizeof(PartialStreamResourceUse))) PartialStreamResourceUse(psru);
						src->next->prev = src;
						src = src->next;
					});

					// splinter the dst uses, and push into the work queue
					difference_one(dst_range, isection, [&](Subrange::Buffer nb) { work_queue.push_back(nb); });

					auto& src_use = *found;

					if (src_use.stream && dst_use.stream && (src_use.stream != dst_use.stream)) {
						dst_use.stream->add_dependency(src_use.stream);
					}
					dst_use.stream->synch_memory(src_use, dst_use, value);

					static_cast<StreamResourceUse&>(*found) = dst_use;
					found->subrange.buffer.offset = isection.offset;
					found->subrange.buffer.size = isection.size;
				}
			}
		}

		StreamResourceUse& last_use(Type* base_ty, void* value) {
			uint64_t key = value_identity(base_ty, value);

			return *last_modify.at(key);
		}
	};

	struct Scheduler : IREvalContext {
		Scheduler(Allocator all, RGCImpl* impl, Recorder& recorder) :
		    allocator(all),
		    recorder(recorder),
		    pass_reads(impl->pass_reads),
		    scheduled_execables(impl->scheduled_execables),
		    impl(impl) {}

		Allocator allocator;
		Recorder& recorder;
		std::pmr::vector<Ref>& pass_reads;
		plf::colony<ScheduledItem>& scheduled_execables;

		InlineArena<std::byte, 4 * 1024> arena;

		RGCImpl* impl;

		size_t naming_index_counter = 0;
		size_t instr_counter = 0;

		void* allocate_host_memory(size_t size) override {
			return arena.ensure_space(size);
		}

		void node_to_acq(Node* node, std::span<void*> values) {
			assert(node->execution_info);
			node->execution_info->kind = node->kind;
			// morph into acquire
			if (node->generic_node.arg_count == (uint8_t)~0u) {
				delete[] node->variable_node.args.data();
			}
			node->kind = Node::ACQUIRE;
			node->acquire = {};

			// initialise storage
			if (!node->acquire.values.data()) { // in case of errors, we might still have the allocation hanging around, we can reuse it
				node->acquire.values = { new void*[node->type.size()], node -> type.size() };
			} else {
				assert(node->acquire.values.size() == node->type.size());
			}
			if (node->rel_acq) {
				node->rel_acq->last_use.resize(node->type.size());
			}

			for (size_t i = 0; i < node->type.size(); i++) {
				auto arg_ty = node->type[i];
				node->acquire.values[i] = new std::byte[arg_ty->size];
				memcpy(node->acquire.values[i], values[i], arg_ty->size);
				auto stripped_ty = Type::stripped(arg_ty);
				if (node->rel_acq) {
					node->rel_acq->last_use[i] = recorder.last_use(stripped_ty.get(), node->acquire.values[i]);
				}
				node->type[i] = stripped_ty;
			}
		}

		template<class T>
		  requires(!std::is_same_v<T, void*> && !std::is_same_v<T, std::span<void*>>)
		void done(Node* node, Stream* stream, T value) {
			auto counter = naming_index_counter;
			naming_index_counter += node->type.size();
			node->execution_info = new (arena.ensure_space(sizeof(ExecutionInfo))) ExecutionInfo{ stream, counter };
			auto value_ptr = static_cast<void*>(new (arena.ensure_space(sizeof(T))) T{ value });
			auto values = new (arena.ensure_space(sizeof(void* [1]))) void*[1];
			values[0] = value_ptr;
			stream->current_submit.push_back(node);
			node_to_acq(node, std::span{ values, 1 });
		}

		void done(Node* node, Stream* stream, void* value_ptr) {
			auto counter = naming_index_counter;
			naming_index_counter += node->type.size();
			node->execution_info = new (arena.ensure_space(sizeof(ExecutionInfo))) ExecutionInfo{ stream, counter };
			stream->current_submit.push_back(node);
			node_to_acq(node, std::span{ &value_ptr, 1 });
		}

		void done(Node* node, Stream* stream, std::span<void*> values) {
			auto counter = naming_index_counter;
			naming_index_counter += node->type.size();
			node->execution_info = new (arena.ensure_space(sizeof(ExecutionInfo))) ExecutionInfo{ stream, counter };
			stream->current_submit.push_back(node);
			node_to_acq(node, values);
		}

		void done(Node* node, Stream* stream) {
			auto counter = naming_index_counter;
			naming_index_counter += node->type.size();
			node->execution_info = new (arena.ensure_space(sizeof(ExecutionInfo))) ExecutionInfo{ stream, counter };
			stream->current_submit.push_back(node);
			assert(node->kind == Node::ACQUIRE);
			node->execution_info->kind = Node::ACQUIRE;
		}

		void fill_render_pass_info(RenderPassInfo& rpass, const size_t& i, CommandBuffer& cobuf) {
			if (rpass.handle == VK_NULL_HANDLE) {
				cobuf.ongoing_render_pass = {};
				return;
			}
			CommandBuffer::RenderPassInfo rpi;
			rpi.render_pass = rpass.handle;
			rpi.subpass = (uint32_t)i;
			rpi.extent = Extent2D{ rpass.fbci.width, rpass.fbci.height };
			auto& spdesc = rpass.rpci.subpass_descriptions[i];
			rpi.color_attachments = std::span<const VkAttachmentReference>(spdesc.pColorAttachments, spdesc.colorAttachmentCount);
			rpi.samples = rpass.fbci.sample_count.count;
			rpi.depth_stencil_attachment = spdesc.pDepthStencilAttachment;
			for (uint32_t i = 0; i < spdesc.colorAttachmentCount; i++) {
				rpi.color_attachment_ivs[i] = rpass.fbci.attachments[i];
			}
			cobuf.color_blend_attachments.resize(spdesc.colorAttachmentCount);
			cobuf.ongoing_render_pass = rpi;
		}

		std::shared_ptr<Type> base_type(Ref parm) {
			return Type::stripped(parm.type());
		}

		std::optional<StreamResourceUse> get_dependency_info(Ref parm, Type* arg_ty, RW type, Stream* dst_stream) {
			auto parm_ty = parm.type();
			auto& link = parm.link();

			std::optional<ResourceUse> s = {};

			if (type == RW::eRead) {
				std::exchange(s, link.read_sync);
			} else {
				std::exchange(s, link.undef_sync);
			}
			if (s) {
				return StreamResourceUse{ *s, dst_stream };
			} else {
				return {};
			}
		}

		Result<void> run() {
			Runtime& ctx = allocator.get_context();
			auto host_stream = recorder.streams.at(DomainFlagBits::eHost).get();

			std::deque<VkPEStream> pe_streams;
			std::unordered_map<uint64_t, Swapchain*> image_to_swapchain;

			Result<void> submit_result = { expected_value };

			for (auto& pitem : impl->item_list) {
				auto& item = *pitem;
				auto node = item.execable;
				instr_counter++;
#ifdef VUK_DUMP_EXEC
				fmt::println("[{:#06x}] {}", instr_counter, exec_to_string(item));
#endif
				// we run nodes twice - first time we reenqueue at the front and then put all deps before it
				// second time we see it, we know that all deps have run, so we can run the node itself
				switch (node->kind) {
				case Node::CONSTANT: {
					done(node, host_stream, node->constant.value);
					break;
				}
				case Node::MATH_BINARY: {
					auto do_op = [&]<class T>(T, Node* node) -> T {
						T& a = *get_value<T>(node->math_binary.a);
						T& b = *get_value<T>(node->math_binary.b);
						switch (node->math_binary.op) {
						case Node::BinOp::ADD:
							return a + b;
						case Node::BinOp::SUB:
							return a - b;
						case Node::BinOp::MUL:
							return a * b;
						case Node::BinOp::DIV:
							return a / b;
						case Node::BinOp::MOD:
							return a % b;
						}
						assert(0);
						return a;
					};
					switch (node->type[0]->kind) {
					case Type::INTEGER_TY: {
						switch (node->type[0]->scalar.width) {
						case 32:
							done(node, host_stream, do_op(uint32_t{}, node));
							break;
						case 64:
							done(node, host_stream, do_op(uint64_t{}, node));
							break;
						default:
							assert(0);
						}
						break;
					}
					default:
						assert(0);
					}
				} break;

				case Node::CONSTRUCT: { // when encountering a CONSTRUCT, construct the thing if needed
					for (auto& arg : node->construct.args) {
						if (arg.node->kind == Node::PLACEHOLDER) {
							return { expected_error, RenderGraphException(format_message(Level::eError, item, "': argument not set or inferrable\n")) };
						}
					}
					// TODO: PAV: use evaluate_construct instead
					assert(node->type[0]->kind != Type::POINTER_TY);
					if (node->type[0]->hash_value == current_module->types.builtin_swapchain) {
						/* no-op */
						done(node, host_stream, get_value(node->construct.args[0]));
						recorder.init_sync(node->type[0].get(), { to_use(eNone), host_stream }, get_value(first(node)));
					} else if (node->type[0]->kind == Type::ARRAY_TY) {
						for (size_t i = 1; i < node->construct.args.size(); i++) {
							auto arg_ty = node->construct.args[i].type();
							auto& parm = node->construct.args[i];

							recorder.add_sync(base_type(parm).get(), get_dependency_info(parm, arg_ty.get(), RW::eWrite, nullptr), get_value(parm));
						}

						auto array_size = node->type[0]->array.count;
						auto elem_ty = *node->type[0]->array.T;
						assert(node->construct.args[0].type()->kind == Type::MEMORY_TY);

						char* arr_mem = static_cast<char*>(arena.ensure_space(elem_ty->size * array_size));
						for (auto i = 0; i < array_size; i++) {
							auto& elem = node->construct.args[i + 1];
							assert(Type::stripped(elem.type())->hash_value == elem_ty->hash_value);

							memcpy(arr_mem + i * elem_ty->size, get_value(elem), elem_ty->size);
						}
						if (array_size == 0) { // zero-len arrays
							arr_mem = nullptr;
						}
						node->construct.args[0].node->constant.value = arr_mem;
						done(node, host_stream, (void*)arr_mem);
					} else if (node->type[0]->hash_value == current_module->types.builtin_sampled_image) {
						for (size_t i = 1; i < node->construct.args.size(); i++) {
							auto arg_ty = node->construct.args[i].type();
							auto& parm = node->construct.args[i];

							recorder.add_sync(base_type(parm).get(), get_dependency_info(parm, arg_ty.get(), RW::eWrite, nullptr), get_value(parm));
						}
						auto image = *get_value<ImageAttachment>(node->construct.args[1]);
						auto samp = *get_value<SamplerCreateInfo>(node->construct.args[2]);
						done(node, host_stream, SampledImage{ image, samp });
					} else if (node->type[0]->kind == Type::UNION_TY) {
						for (size_t i = 1; i < node->construct.args.size(); i++) {
							auto arg_ty = node->construct.args[i].type();
							auto& parm = node->construct.args[i];

							recorder.add_sync(base_type(parm).get(), get_dependency_info(parm, arg_ty.get(), RW::eWrite, nullptr), get_value(parm));
						}
						assert(node->construct.args[0].type()->kind == Type::MEMORY_TY);

						char* arr_mem = static_cast<char*>(arena.ensure_space(node->type[0]->size));
						size_t offset = 0;
						for (auto i = 0; i < node->construct.args.size() - 1; i++) {
							auto sz = node->type[0]->composite.types[i]->size;
							auto& elem = node->construct.args[i + 1];
							memcpy(arr_mem + offset, get_value(elem), sz);
							offset += sz;
						}

						node->construct.args[0].node->constant.value = arr_mem;
						done(node, host_stream, (void*)arr_mem);
					} else {
						for (size_t i = 1; i < node->construct.args.size(); i++) {
							auto arg_ty = node->construct.args[i].type();
							auto& parm = node->construct.args[i];

							recorder.add_sync(base_type(parm).get(), get_dependency_info(parm, arg_ty.get(), RW::eWrite, nullptr), get_value(parm));
						}

						auto result_ty = node->type[0].get();
						// allocate type
						void* result = new char[result_ty->size];
						// loop args and resolve them
						std::vector<void*> argvals;
						for (size_t i = 1; i < node->construct.args.size(); i++) {
							auto& parm = node->construct.args[i];
							argvals.push_back(get_value(parm));
						}

						result_ty->composite.construct(result, argvals);
						// TODO: PAV: user type sync
						recorder.init_sync(node->type[0].get(), { to_use(eNone), host_stream }, result,
						                   false); // TODO: can we figure out when it is safe known aliasing?
						done(node, host_stream, result);
					}

					break;
				}

				// we can allocate ptrs and generic views
				// TODO: image ptrs and generic views
				case Node::ALLOCATE: {
					auto allocator = node->allocate.allocator ? *node->allocate.allocator : this->allocator;

					if (node->type[0]->kind == Type::POINTER_TY) {
						auto pointed_ty = *node->type[0]->pointer.T;

						ptr_base buf;
						auto bci = *get_value<BufferCreateInfo>(node->allocate.src);
						if (auto res = allocator.allocate_memory(std::span{ static_cast<ptr_base*>(&buf), 1 }, std::span{ &bci, 1 }); !res) {
							return res;
						}
						allocator.deallocate(std::span{ static_cast<ptr_base*>(&buf), 1 });
						done(node, host_stream, buf);
					} else if (node->type[0]->hash_value == current_module->types.builtin_image) {
						auto& attachment = constant<ImageAttachment>(node->construct.args[0]);
						// set iv type
						if (attachment.image_view == ImageView{}) {
							if (attachment.view_type == ImageViewType::eInfer && attachment.layer_count != VK_REMAINING_ARRAY_LAYERS) {
								if (attachment.image_type == ImageType::e1D) {
									if (attachment.layer_count == 1) {
										attachment.view_type = ImageViewType::e1D;
									} else {
										attachment.view_type = ImageViewType::e1DArray;
									}
								} else if (attachment.image_type == ImageType::e2D) {
									if (attachment.layer_count == 1) {
										attachment.view_type = ImageViewType::e2D;
									} else {
										attachment.view_type = ImageViewType::e2DArray;
									}
								} else if (attachment.image_type == ImageType::e3D) {
									if (attachment.layer_count == 1) {
										attachment.view_type = ImageViewType::e3D;
									} else {
										attachment.view_type = ImageViewType::e2DArray;
									}
								}
							}
						}
						if (!attachment.image) {
							attachment.usage |= impl->compute_usage(&first(node).link());
							assert(attachment.usage != ImageUsageFlags{});
							auto img = allocate_image(allocator, attachment);
							if (!img) {
								return img;
							}
							attachment.image = **img;
							if (node->debug_info && node->debug_info->result_names.size() > 0 && !node->debug_info->result_names[0].empty()) {
								ctx.set_name(attachment.image.image, node->debug_info->result_names[0].c_str());
							}
						}
						done(node, host_stream, attachment);
					} else {
						assert(false); // nothing else can be allocated
					}
					recorder.init_sync(node->type[0].get(), { to_use(eNone), host_stream }, get_value(first(node)));

					break;
				}

				case Node::CALL: {
					auto fn_type = node->call.args[0].type();
					size_t first_parm = fn_type->kind == Type::OPAQUE_FN_TY ? 1 : 4;
					auto& args = fn_type->kind == Type::OPAQUE_FN_TY ? fn_type->opaque_fn.args : fn_type->shader_fn.args;

					Stream* dst_stream = item.scheduled_stream; // the domain this call will execute on

					auto vk_rec = dynamic_cast<VkQueueStream*>(dst_stream); // TODO: change this into dynamic dispatch on the Stream
					assert(vk_rec);
					// run all the barriers here!

					for (size_t i = first_parm; i < node->call.args.size(); i++) {
						auto& arg_ty = args[i - first_parm];
						auto& parm = node->call.args[i];
						auto& link = parm.link();

						if (arg_ty->kind == Type::IMBUED_TY) {
							auto access = arg_ty->imbued.access;

							// here: figuring out which allocator to use to make image views for the RP and then making them
							if (is_framebuffer_attachment(access)) {
								auto& img_att = *get_value<ImageAttachment>(parm);
								if (img_att.view_type == ImageViewType::eInfer || img_att.view_type == ImageViewType::eCube) { // framebuffers need 2D or 2DArray views
									if (img_att.layer_count > 1) {
										img_att.view_type = ImageViewType::e2DArray;
									} else {
										img_att.view_type = ImageViewType::e2D;
									}
								}
								if (img_att.image_view.payload == VK_NULL_HANDLE) {
									auto iv = allocate_image_view(allocator, img_att); // TODO: dropping error
									img_att.image_view = **iv;

									auto name = std::string("ImageView: RenderTarget ");
									allocator.get_context().set_name(img_att.image_view.payload, Name(name));
								}
							}

							// Write and ReadWrite
							RW sync_access = (is_write_access(access)) ? RW::eWrite : RW::eRead;
							recorder.add_sync(base_type(parm).get(), get_dependency_info(parm, arg_ty.get(), sync_access, dst_stream), get_value(parm));

							if (is_framebuffer_attachment(access)) {
								auto& img_att = *get_value<ImageAttachment>(parm);
								vk_rec->prepare_render_pass_attachment(allocator, img_att);
							}
						} else {
							assert(0);
						}
					}

					// make the renderpass if needed!
					recorder.synchronize_stream(dst_stream);
					// run the user cb!
					std::vector<void*, short_alloc<void*>> opaque_rets(*impl->arena_);
					if (fn_type->kind == Type::OPAQUE_FN_TY) {
						CommandBuffer cobuf(*dst_stream, ctx, allocator, vk_rec->cbuf);
						if (!fn_type->debug_info.name.empty()) {
							auto name_hash = static_cast<uint32_t>(std::hash<std::string>{}(fn_type->debug_info.name));
							auto name_color = std::array<float, 4>{
								static_cast<float>(name_hash & 255) / 255.0f,
								static_cast<float>((name_hash >> 8) & 255) / 255.0f,
								static_cast<float>((name_hash >> 16) & 255) / 255.0f,
								1.0,
							};
							ctx.begin_region(vk_rec->cbuf, fn_type->debug_info.name.c_str(), name_color);
						}

						void* rpass_profile_data = nullptr;
						if (vk_rec->callbacks->on_begin_pass)
							rpass_profile_data = vk_rec->callbacks->on_begin_pass(vk_rec->callbacks->user_data, fn_type->debug_info.name.c_str(), cobuf, vk_rec->domain);

						if (vk_rec->rp.rpci.attachments.size() > 0) {
							vk_rec->prepare_render_pass();
							fill_render_pass_info(vk_rec->rp, 0, cobuf);
						}

						std::vector<void*, short_alloc<void*>> opaque_args(*impl->arena_);
						std::vector<void*, short_alloc<void*>> opaque_meta(*impl->arena_);
						for (size_t i = first_parm; i < node->call.args.size(); i++) {
							auto& parm = node->call.args[i];
							opaque_args.push_back(get_value(parm));
							opaque_meta.push_back(&parm);
						}
						opaque_rets.resize(fn_type->opaque_fn.return_types.size());
						(*fn_type->callback)(cobuf, opaque_args, opaque_meta, opaque_rets);
						if (vk_rec->rp.handle) {
							vk_rec->end_render_pass();
						}
						if (!fn_type->debug_info.name.empty()) {
							ctx.end_region(vk_rec->cbuf);
						}
						if (vk_rec->callbacks->on_end_pass)
							vk_rec->callbacks->on_end_pass(vk_rec->callbacks->user_data, rpass_profile_data, cobuf);
					} else if (fn_type->kind == Type::SHADER_FN_TY) {
						CommandBuffer cobuf(*dst_stream, ctx, allocator, vk_rec->cbuf);
						if (!fn_type->debug_info.name.empty()) {
							auto name_hash = static_cast<uint32_t>(std::hash<std::string>{}(fn_type->debug_info.name));
							auto name_color = std::array<float, 4>{
								static_cast<float>(name_hash & 255) / 255.0f,
								static_cast<float>((name_hash >> 8) & 255) / 255.0f,
								static_cast<float>((name_hash >> 16) & 255) / 255.0f,
								1.0,
							};
							ctx.begin_region(vk_rec->cbuf, fn_type->debug_info.name.c_str(), name_color);
						}

						void* rpass_profile_data = nullptr;
						if (vk_rec->callbacks->on_begin_pass)
							rpass_profile_data = vk_rec->callbacks->on_begin_pass(vk_rec->callbacks->user_data, fn_type->debug_info.name.c_str(), cobuf, vk_rec->domain);

						if (vk_rec->rp.rpci.attachments.size() > 0) {
							vk_rec->prepare_render_pass();
							fill_render_pass_info(vk_rec->rp, 0, cobuf);
						}

						// call the cbuf directly: bind everything, then dispatch shader
						opaque_rets.resize(fn_type->shader_fn.return_types.size());
						auto pbi = reinterpret_cast<PipelineBaseInfo*>(fn_type->shader_fn.shader);

						cobuf.bind_compute_pipeline(pbi);

						auto& flat_bindings = pbi->reflection_info.flat_bindings;
						for (size_t i = first_parm; i < node->call.args.size(); i++) {
							auto& parm = node->call.args[i];
							if (parm.type()->kind != Type::POINTER_TY) {
								auto binding_idx = i - first_parm;
								auto& [set, binding] = flat_bindings[binding_idx];
								auto val = get_value(parm);
								switch (binding->type) {
								case DescriptorType::eSampledImage:
								case DescriptorType::eStorageImage:
									cobuf.bind_image(set, binding->binding, *reinterpret_cast<ImageAttachment*>(val));
									break;
								case DescriptorType::eUniformBuffer:
								case DescriptorType::eStorageBuffer: {
									auto& v = *reinterpret_cast<Buffer<>*>(val);
									cobuf.bind_buffer(set, binding->binding, v);
									break;
								}
								case DescriptorType::eSampler:
									cobuf.bind_sampler(set, binding->binding, *reinterpret_cast<SamplerCreateInfo*>(val));
									break;
								case DescriptorType::eCombinedImageSampler: {
									auto& si = *reinterpret_cast<SampledImage*>(val);
									cobuf.bind_image(set, binding->binding, si.ia);
									cobuf.bind_sampler(set, binding->binding, si.sci);
									break;
								}
								default:
									assert(0);
								}

								opaque_rets[binding_idx] = val;
							}
						}
						size_t pc_offset = 0;
						if (pbi->reflection_info.push_constant_ranges.size() > 0) {
							auto& pcr = pbi->reflection_info.push_constant_ranges[0];
							auto base_ty = current_module->types.make_pointer_ty(current_module->types.u32());
							for (auto parm_idx = 0; parm_idx < pcr.num_members; parm_idx++) {
								auto& parm = node->call.args[parm_idx + first_parm];
								auto val = get_value(parm);
								auto ptr = *reinterpret_cast<ptr_base*>(val);
								// TODO: check which args are pointers and dereference on host the once that are not
								cobuf.push_constants(ShaderStageFlagBits::eCompute, pc_offset, ptr);
								auto binding_idx = parm_idx;
								opaque_rets[binding_idx] = val;
								parm_idx++;
								pc_offset += sizeof(uint64_t);
							}
						}

						cobuf.dispatch(constant<uint32_t>(node->call.args[1]), constant<uint32_t>(node->call.args[2]), constant<uint32_t>(node->call.args[3]));

						if (vk_rec->rp.handle) {
							vk_rec->end_render_pass();
						}
						if (!fn_type->debug_info.name.empty()) {
							ctx.end_region(vk_rec->cbuf);
						}
						if (vk_rec->callbacks->on_end_pass)
							vk_rec->callbacks->on_end_pass(vk_rec->callbacks->user_data, rpass_profile_data, cobuf);
					} else {
						assert(0);
					}

					done(node, dst_stream, std::span(opaque_rets));

					break;
				}
				case Node::RELEASE: {
					auto acqrel = node->rel_acq;

					assert(acqrel && acqrel->status == Signal::Status::eDisarmed);

					Stream* dst_stream;
					Swapchain* swp = nullptr;
					if (node->release.dst_domain == DomainFlagBits::ePE) {
						swp = reinterpret_cast<Swapchain*>(
						    image_to_swapchain.at(recorder.value_identity(node->release.src[0].type().get(), get_value(node->release.src[0]))));
						auto it = std::find_if(pe_streams.begin(), pe_streams.end(), [=](auto& pe_stream) { return pe_stream.swp == swp; });
						assert(it != pe_streams.end());
						dst_stream = &*it;
					} else if (node->release.dst_domain == DomainFlagBits::eDevice) {
						dst_stream = item.scheduled_stream;
					} else {
						dst_stream = recorder.stream_for_domain(node->release.dst_domain);
					}
					assert(dst_stream);

					auto sched_stream = item.scheduled_stream;
					DomainFlagBits sched_domain = sched_stream->domain;
					DomainFlagBits dst_domain = dst_stream->domain;

					node->rel_acq->last_use.resize(node->type.size());
					auto values = new (arena.ensure_space(sizeof(void*) * node->type.size())) void*[node->type.size()];

					for (size_t i = 0; i < node->release.src.size(); i++) {
						auto parm = node->release.src[i];
						auto arg_ty = node->type[i];
						auto di = get_dependency_info(parm, arg_ty.get(), RW::eWrite, dst_stream);
						auto value = get_value(parm);
						values[i] = value;
						recorder.add_sync(base_type(parm).get(), di, value);

						auto last_use = recorder.last_use(base_type(parm).get(), value);
						// SANITY: if we change streams, then we must've had sync
						// TODO: remove host exception here
						assert(di || last_use.stream->domain == DomainFlagBits::eHost || (last_use.stream == item.scheduled_stream));
						acqrel->last_use.push_back(last_use);
						if (i == 0) {
							sched_stream->add_dependent_signal(acqrel);
						}
					}

					if (acqrel && sched_domain == DomainFlagBits::eHost) {
						acqrel->status = Signal::Status::eHostAvailable;
					}

					if (dst_domain == DomainFlagBits::ePE) {
						assert(sched_stream->domain & DomainFlagBits::eDevice);
						assert(swp);
						auto present_result = dynamic_cast<VkQueueStream*>(sched_stream)->present(*swp);
						if (!present_result) {
							submit_result = std::move(present_result);
						}
						if (acqrel) {
							acqrel->status = Signal::Status::eHostAvailable; // TODO: ???
						}
					} else {
						sched_stream->submit();
					}
					host_stream->submit();

					done(node, item.scheduled_stream, std::span(values, node->type.size()));
					break;
				}
				case Node::ACQUIRE: {
					auto acqrel = node->rel_acq;
					assert(acqrel && acqrel->status != Signal::Status::eDisarmed);

					auto src_stream = acqrel->source.executor ? recorder.stream_for_executor(acqrel->source.executor) : recorder.stream_for_domain(DomainFlagBits::eHost);
					for (size_t i = 0; i < node->acquire.values.size(); i++) {
						auto& link = node->links[i];

						StreamResourceUse src_use = { acqrel->last_use[i], src_stream };
						recorder.init_sync(node->type[i].get(), src_use, node->acquire.values[i], false);
					}

					done(node, src_stream);
					break;
				}

				case Node::ACQUIRE_NEXT_IMAGE: {
					auto& swp = **get_value<Swapchain*>(node->acquire_next_image.swapchain);
					VkSemaphore acquire_sema;
					allocator.allocate_semaphores(std::span{ &acquire_sema, 1 });
					allocator.deallocate(std::span{ &acquire_sema, 1 });
					swp.acquire_result = ctx.vkAcquireNextImageKHR(ctx.device, swp.swapchain, UINT64_MAX, acquire_sema, VK_NULL_HANDLE, &swp.image_index);
					// VK_SUBOPTIMAL_KHR shouldn't stop presentation; it is handled at the end
					if (swp.acquire_result != VK_SUCCESS && swp.acquire_result != VK_SUBOPTIMAL_KHR) {
						return { expected_error, VkException{ swp.acquire_result } };
					}

					auto pe_stream = &pe_streams.emplace_back(allocator, swp, acquire_sema);
					done(node, pe_stream, swp.images[swp.image_index]);
					image_to_swapchain.emplace(recorder.value_identity(node->type[0].get(), &swp.images[swp.image_index]), &swp);
					auto& lu = recorder.last_use(node->type[0].get(), &swp.images[swp.image_index]);
					lu = StreamResourceUse{ { PipelineStageFlagBits::eAllCommands, AccessFlagBits::eNone, ImageLayout::eUndefined }, pe_stream };

					break;
				}
				case Node::SLICE: {
					// half sync
					recorder.add_sync(base_type(node->slice.src).get(),
					                  get_dependency_info(node->slice.src, node->slice.src.type().get(), RW::eRead, item.scheduled_stream),
					                  get_value(node->slice.src));
					auto composite = node->slice.src;
					void* composite_v = get_value(composite);
					auto axis = node->slice.axis;
					auto start = *get_value<uint64_t>(node->slice.start);
					auto count = *get_value<uint64_t>(node->slice.count);
					auto t = Type::stripped(composite.type());

					if (!(node->debug_info && node->debug_info->result_names.size() > 0 && !node->debug_info->result_names[0].empty())) {
						/*std::string name = fmt::format("{}_{}[{}->{}:{}]",
						                               Node::kind_to_sv(node->slice.src.node->execution_info->kind),
						                               node->slice.src.node->execution_info->naming_index,
						                               node->slice.axis,
						                               start,
						                               start + count - 1);
						current_module->name_output(first(node), name);*/
					}
					std::vector<void*, short_alloc<void*>> rets(3, *impl->arena_);
					rets[0] = impl->arena_->allocate(node->type[0]->size);
					evaluate_slice(composite, axis, start, count, composite_v, rets[0]);
					rets[1] = impl->arena_->allocate(node->slice.src.type()->size);
					memcpy(rets[1], get_value(node->slice.src), node->slice.src.type()->size);
					rets[2] = impl->arena_->allocate(node->slice.src.type()->size);
					memcpy(rets[2], get_value(node->slice.src), node->slice.src.type()->size);
					done(node, node->slice.src.node->execution_info->stream, std::span(rets));

					break;
				}
				case Node::CONVERGE: {
					auto base = node->converge.diverged[0];

					// half sync
					for (size_t i = 0; i < node->converge.diverged.size(); i++) {
						auto& div = node->converge.diverged[i];
						recorder.add_sync(base_type(div).get(), get_dependency_info(div, div.type().get(), RW::eWrite, base.node->execution_info->stream), get_value(div));
					}

					done(node, base.node->execution_info->stream, get_value(base));
					break;
				}
				case Node::USE: {
					// half sync
					auto& div = node->use.src;
					recorder.add_sync(base_type(div).get(), get_dependency_info(div, div.type().get(), RW::eWrite, div.node->execution_info->stream), get_value(div));

					done(node, div.node->execution_info->stream, get_value(div));

					break;
				}
				case Node::LOGICAL_COPY: {
					// half sync
					auto& div = node->logical_copy.src;
					recorder.add_sync(base_type(div).get(), get_dependency_info(div, div.type().get(), RW::eWrite, div.node->execution_info->stream), get_value(div));

					done(node, div.node->execution_info->stream, get_value(div));

					break;
				}
				case Node::COMPILE_PIPELINE: {
					auto& src = node->compile_pipeline.src;
					auto& pbci = *get_value<PipelineBaseCreateInfo>(src);
					auto pipeline = allocator.get_context().get_pipeline(pbci);

					done(node, host_stream, pipeline);
					break;
				}
				case Node::GET_ALLOCATION_SIZE: {
					auto ptr = *get_value<ptr_base>(node->get_allocation_size.ptr);
					auto size = allocator.get_context().resolve_ptr(ptr).buffer.size;

					done(node, item.scheduled_stream, size);
					break;
				}
				default:
					assert(0);
				}
			}
			return submit_result;
		}
	};

	Result<void> Compiler::execute(Allocator& alloc) {
		Runtime& ctx = alloc.get_context();

		Recorder recorder(alloc, &impl->callbacks, impl->pass_reads);
		recorder.streams.emplace(DomainFlagBits::eHost, std::make_unique<HostStream>(alloc));
		if (auto exe = ctx.get_executor(DomainFlagBits::eGraphicsQueue)) {
			recorder.streams.emplace(DomainFlagBits::eGraphicsQueue, std::make_unique<VkQueueStream>(alloc, static_cast<QueueExecutor*>(exe), &impl->callbacks));
		}
		if (auto exe = ctx.get_executor(DomainFlagBits::eComputeQueue)) {
			recorder.streams.emplace(DomainFlagBits::eComputeQueue, std::make_unique<VkQueueStream>(alloc, static_cast<QueueExecutor*>(exe), &impl->callbacks));
		}
		if (auto exe = ctx.get_executor(DomainFlagBits::eTransferQueue)) {
			recorder.streams.emplace(DomainFlagBits::eTransferQueue, std::make_unique<VkQueueStream>(alloc, static_cast<QueueExecutor*>(exe), &impl->callbacks));
		}
		auto host_stream = recorder.streams.at(DomainFlagBits::eHost).get();
		host_stream->executor = ctx.get_executor(DomainFlagBits::eHost);
		recorder.last_modify.at(0)->stream = host_stream;

		for (auto& item : impl->item_list) {
			item->scheduled_stream = recorder.stream_for_domain(item->scheduled_domain);
			if (!item->scheduled_stream && item->scheduled_domain != DomainFlagBits::eNone) {
				return { expected_error,
					       RenderGraphException(
					           format_message(Level::eError,
					                          *item,
					                          fmt::format("': requested stream from Domain<{}>, but the Runtime was not provided an Executor for this Domain\n",
					                                      domain_to_string(item->scheduled_domain)))) };
			}
		}

		Scheduler sched(alloc, impl, recorder);

		auto submit_result = sched.run();
		if (!submit_result) {
			return submit_result;
		}

		// post-run: checks and cleanup
		std::vector<std::shared_ptr<IRModule>> modules;
		for (auto& depnode : impl->depnodes) {
			modules.push_back(depnode->source_module);
		}
		std::sort(modules.begin(), modules.end());
		modules.erase(std::unique(modules.begin(), modules.end()), modules.end());

		impl->depnodes.clear();

		// populate values and last_use
		for (auto& [def_link, lr] : impl->live_ranges) {
			assert(def_link);
			assert(lr.undef_link);
			if (def_link->def.node->kind == Node::CONSTANT) {
				continue;
			}

			// get final value
			Ref final_use = lr.undef_link->def;
			assert(!final_use.node->rel_acq || final_use.node->rel_acq->status != Signal::Status::eDisarmed);
			lr.last_value = get_value(final_use);
			lr.last_use = recorder.last_use(Type::stripped(final_use.type()).get(), lr.last_value);

			// get final signal
			AcquireRelease* last_signal = nullptr;
			for (auto link = lr.undef_link; link; link = link->prev) {
				if (link->def.node->rel_acq) {
					last_signal = link->def.node->rel_acq;
					break;
				}
			}

			// put the values on the nodes
			for (auto link = def_link; link; link = link->next) {
				auto& ref = link->def;
				assert(ref);
				assert(ref.node->kind == Node::ACQUIRE);
				memcpy(ref.node->acquire.values[ref.index], lr.last_value, ref.node->type[ref.index]->size);
				if (ref.node->rel_acq) {
					ref.node->rel_acq->last_use[ref.index] = lr.last_use;
				}
				if (ref.node->rel_acq && last_signal) {
					ref.node->rel_acq->source = last_signal->source;
					ref.node->rel_acq->status = last_signal->status;
				}
			}
		}

		for (auto& node : impl->nodes) {
			// shrink slice acquires
			if (node->execution_info && node->execution_info->kind == Node::SLICE && node->rel_acq) {
				for (size_t i = 1; i < node->acquire.values.size(); i++) {
					current_module->types.destroy(Type::stripped(node->type[i]).get(), node->acquire.values[i]);
				}
				node->acquire.values = { node->acquire.values.data(), 1 };
				node->type = { node->type.data(), 1 };
			}

			// reset any nodes we ran
			node->execution_info = nullptr;
			node->links = nullptr;
			node->scheduled_item = nullptr;
		}

		impl->garbage_nodes.insert(impl->garbage_nodes.end(), current_module->garbage.begin(), current_module->garbage.end());
		std::sort(impl->garbage_nodes.begin(), impl->garbage_nodes.end());
		impl->garbage_nodes.erase(std::unique(impl->garbage_nodes.begin(), impl->garbage_nodes.end()), impl->garbage_nodes.end());
		for (auto& node : impl->garbage_nodes) {
			current_module->destroy_node(node);
		}

		current_module->garbage.clear();
		impl->garbage_nodes.clear();

		for (auto& m : modules) {
			for (auto& op : m->op_arena) {
				op.links = nullptr;
			}
		}

		current_module->types.collect();

		return submit_result;
	} // namespace vuk
} // namespace vuk
