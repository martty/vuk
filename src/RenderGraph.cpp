#include "RenderGraph.hpp"
#include "Hash.hpp" // for create
#include "Cache.hpp"
#include "Context.hpp"
#include "CommandBuffer.hpp"
#include "Allocator.hpp"

namespace vuk {
	// determine rendergraph inputs and outputs, and resources that are neither
	void RenderGraph::build_io() {
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

	void RenderGraph::build() {
		// compute sync
		// find which reads are graph inputs (not produced by any pass) & outputs (not consumed by any pass)
		build_io();
		// sort passes
		if (passes.size() > 1) {
			topological_sort(passes.begin(), passes.end(), [](const auto& p1, const auto& p2) {
				bool could_execute_after = false;
				bool could_execute_before = false;
				for (auto& o : p1.outputs) {
					if (p2.inputs.contains(o)) could_execute_after = true;
				}
				for (auto& o : p2.outputs) {
					if (p1.inputs.contains(o)) could_execute_before = true;
				}
				if (could_execute_after && could_execute_before) {
					return p1.pass.auxiliary_order < p2.pass.auxiliary_order;
				} else if (could_execute_after) {
					return true;
				} else 
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
				[&](::Buffer& th) {
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
							write_access = (th.type == ::Buffer::Type::eStorage || th.type == ::Buffer::Type::eUniform) ?
								vk::AccessFlagBits::eShaderWrite : vk::AccessFlagBits::eTransferWrite;
							write_stage = (th.type == ::Buffer::Type::eStorage || th.type == ::Buffer::Type::eUniform) ?
								vk::PipelineStageFlagBits::eAllGraphics : vk::PipelineStageFlagBits::eTransfer;
						}

						if (contains(p.pass.read_buffers, th)) {
							if (!dst)
								dst = &p;
							// handle a single type of dst queue for now
							assert(read_queue == "INVALID" || read_queue == p.pass.executes_on);
							read_queue = p.pass.executes_on;
							read_access |= (th.type == ::Buffer::Type::eStorage || th.type == ::Buffer::Type::eUniform) ?
								vk::AccessFlagBits::eShaderRead : vk::AccessFlagBits::eTransferRead;
							read_stage |= (th.type == ::Buffer::Type::eStorage || th.type == ::Buffer::Type::eUniform) ?
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
			if (passinfo.pass.depth_attachment) atts.insert(*passinfo.pass.depth_attachment);

			if (auto p = contains_if(attachment_sets, [&](auto& t) { return t.first == atts; })) {
				p->second.push_back(&passinfo);
			} else {
				attachment_sets.emplace_back(atts, std::vector{ &passinfo });
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
					si.attachments.emplace(a.name, AttachmentSInfo{ vk::ImageLayout::eColorAttachmentOptimal });
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
				if (p->pass.depth_attachment) {
					auto& a = *p->pass.depth_attachment;
					si.attachments.emplace(a.name, AttachmentSInfo{ vk::ImageLayout::eDepthStencilAttachmentOptimal });
					// TODO: ColorAttachmentRead happens if blending or logicOp
					if (contains_if(rpi.attachments, [&](auto& att) { return att.name == a.name; })) {
						// TODO: we want to add all same level passes here, otherwise the other passes might not get the right sync
					} else {
						AttachmentRPInfo arpi;
						arpi.name = a.name;
						arpi.first_use.access = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
						arpi.first_use.stage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
						arpi.first_use.subpass = subpass - 1;

						rpi.attachments.push_back(arpi);
					}

				}
				rpi.subpasses.push_back(si);
			}
			rpis.push_back(rpi);
		}

	}

	void RenderGraph::bind_attachment_to_swapchain(Name name, vk::Format format, vk::Extent2D extent, vk::ImageView siv) {
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

		attachment_info.is_external = true;
		// for WSI, we want to wait for colourattachmentoutput
		attachment_info.srcStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
		// we don't care about any writes, we will clear
		attachment_info.srcAccess = vk::AccessFlags{};
		bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::mark_attachment_internal(Name name, vk::Format format, vk::Extent2D extent) {
		AttachmentRPInfo attachment_info;
		attachment_info.extents = extent;
		attachment_info.iv = vk::ImageView{};
		attachment_info.is_external = false;
		// internal attachment, discard contents
		attachment_info.description.initialLayout = vk::ImageLayout::eUndefined;
		
		//attachment_info.description.finalLayout = vk::ImageLayout::ePresentSrcKHR;
		attachment_info.description.loadOp = vk::AttachmentLoadOp::eClear;
		attachment_info.description.storeOp = vk::AttachmentStoreOp::eDontCare;

		attachment_info.description.format = format;

		bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::build(vuk::PerThreadContext& ptc) {
		// output attachments have an initial layout of eUndefined (loadOp: clear or dontcare, storeOp: store)
		for (auto& [name, attachment_info] : bound_attachments) {
			if (!attachment_info.is_external) continue;
			// get the first pass this attachment is used
			auto pass = contains_if(passes, [&](auto& pi) {
				auto& p = pi.pass;
				if (contains_if(p.color_attachments, [&](auto& att) { return att.name == name; })) return true;
				if (p.depth_attachment && p.depth_attachment->name == name) return true;
				return false;
				});
			// in the renderpass this pass participates in, we want to fill in the info for the attachment
			// TODO: we want to find the first to put in the initial layout
			// then we want to find the last to put in the final layout
			assert(pass);
			auto& dst = *contains_if(rpis[pass->render_pass_index].attachments, [&](auto& att) {return att.name == name; });
			dst.description = attachment_info.description;
			dst.srcAccess = attachment_info.srcAccess;
			dst.srcStage = attachment_info.srcStage;
			dst.iv = attachment_info.iv;
			dst.extents = attachment_info.extents;
			dst.is_external = true;
		}
		// do this for input attachments
		// input attachments have a finallayout matching last use
		// loadOp: load, storeOp: dontcare
		// inout attachments
		// loadOp: load, storeOp: store

		// -- transient fb attachments --
		// perform allocation
		for (auto& [name, attachment_info] : bound_attachments) {
			if (attachment_info.is_external) continue;
			// get the first pass this attachment is used
			auto pass = contains_if(passes, [&](auto& pi) {
				auto& p = pi.pass;
				if (contains_if(p.color_attachments, [&](auto& att) { return att.name == name; })) return true;
				if (p.depth_attachment && p.depth_attachment->name == name) return true;
				return false;
				});
			assert(pass);
			auto& dst = *contains_if(rpis[pass->render_pass_index].attachments, [&](auto& att) {return att.name == name; });
			dst.description = attachment_info.description;
			// TODO: this should match last use
			dst.description.finalLayout = vk::ImageLayout::eGeneral;
			dst.extents = attachment_info.extents;
			dst.is_external = false;
			vk::ImageCreateInfo ici;
			ici.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
			ici.arrayLayers = 1;
			ici.extent = vk::Extent3D(dst.extents, 1);
			ici.imageType = vk::ImageType::e2D;
			ici.format = vk::Format::eD32Sfloat;
			ici.mipLevels = 1;
			ici.initialLayout = vk::ImageLayout::eUndefined;
			ici.samples = vk::SampleCountFlagBits::e1;
			ici.sharingMode = vk::SharingMode::eExclusive;
			ici.tiling = vk::ImageTiling::eOptimal;

			vk::ImageViewCreateInfo ivci;
			ivci.image = vk::Image{};
			ivci.format = dst.description.format;
			ivci.viewType = vk::ImageViewType::e2D;
			vk::ImageSubresourceRange isr;
			isr.aspectMask = vk::ImageAspectFlagBits::eDepth;
			isr.baseArrayLayer = 0;
			isr.layerCount = 1;
			isr.baseMipLevel = 0;
			isr.levelCount = 1;
			ivci.subresourceRange = isr;

			RGCI rgci;
			rgci.ici = ici;
			rgci.ivci = ivci;
			
			auto rg = ptc.transient_images.acquire(rgci);
			dst.iv = rg.image_view;
		}

		// we now have enough data to build vk::RenderPasses and vk::Framebuffers
		for (auto& rp : rpis) {
			// subpasses
			auto& subp = rp.rpci.subpass_descriptions;
			auto& color_attrefs = rp.rpci.color_refs;
			auto& color_ref_offsets = rp.rpci.color_ref_offsets;
			auto& ds_attrefs = rp.rpci.ds_refs;
			ds_attrefs.resize(rp.subpasses.size());
			for (size_t i = 0; i < rp.subpasses.size(); i++) {
				auto& s = rp.subpasses[i];
				for (auto& [name, att] : s.attachments) {
					vk::AttachmentReference attref;
					attref.layout = att.layout;
					attref.attachment = std::distance(rp.attachments.begin(), std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) { return name == att.name; }));

					if (att.layout != vk::ImageLayout::eColorAttachmentOptimal) {
						if (att.layout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
							ds_attrefs[i] = attref;
						}
					} else {
						color_attrefs.push_back(attref);
					}
				}
				color_ref_offsets.push_back(color_attrefs.size());
			}
			for (size_t i = 0; i < rp.subpasses.size(); i++) {
				vuk::SubpassDescription sd;
				sd.colorAttachmentCount = color_ref_offsets[i] - (i > 0 ? color_ref_offsets[i - 1] : 0);
				sd.pColorAttachments = color_attrefs.data() + (i > 0 ? color_ref_offsets[i - 1] : 0);
				sd.pDepthStencilAttachment = ds_attrefs[i] ? &*ds_attrefs[i] : nullptr;
				sd.flags = {};
				sd.inputAttachmentCount = 0;
				sd.pInputAttachments = nullptr;
				sd.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
				sd.preserveAttachmentCount = 0;
				sd.pPreserveAttachments = nullptr;
				sd.pResolveAttachments = nullptr;

				subp.push_back(sd);
			}

			rp.rpci.pSubpasses = rp.rpci.subpass_descriptions.data();
			rp.rpci.subpassCount = rp.rpci.subpass_descriptions.size();
			// generate external deps
			auto& deps = rp.rpci.subpass_dependencies;
			for (auto& s : rp.subpasses) {
				if (s.pass->is_head_pass) {
					for (auto& attrpinfo : rp.attachments) {
						if (attrpinfo.first_use.subpass == s.pass->subpass && attrpinfo.is_external) {
							vk::SubpassDependency dep_in;
							dep_in.srcSubpass = VK_SUBPASS_EXTERNAL;
							dep_in.dstSubpass = s.pass->subpass;

							dep_in.srcAccessMask = attrpinfo.srcAccess;
							dep_in.srcStageMask = attrpinfo.srcStage;
							dep_in.dstAccessMask = attrpinfo.first_use.access;
							dep_in.dstStageMask = attrpinfo.first_use.stage;

							deps.push_back(dep_in);
						}
					}
				}
			}
			// subpass-subpass deps
			//
			for (auto& attrpinfo : rp.attachments) {
				size_t last_subpass_use_index = -1;
				for (size_t i = 0; i < rp.subpasses.size(); i++) {
					auto& s = rp.subpasses[i];
					if(s.pass->inputs.contains(attrpinfo.name)){
						if (last_subpass_use_index == -1) {
							last_subpass_use_index = i;
							continue;
						}
						auto& last_subpass = rp.subpasses[last_subpass_use_index];
						auto& satt = last_subpass.attachments[attrpinfo.name];
						auto& catt = s.attachments[attrpinfo.name];
						vk::SubpassDependency dep;
						dep.srcSubpass = last_subpass_use_index;
						dep.dstSubpass = i;
						if (satt.layout == vk::ImageLayout::eColorAttachmentOptimal) {
							dep.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead;
							dep.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
						} else if (satt.layout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
							dep.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
							dep.srcStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests;
						}
						if (catt.layout == vk::ImageLayout::eColorAttachmentOptimal) {
							dep.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead;
							dep.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
						} else if (catt.layout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
							dep.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
							dep.dstStageMask = vk::PipelineStageFlagBits::eEarlyFragmentTests;
						}

						deps.push_back(dep);
						last_subpass_use_index = i;
					}
				}
			}

			// TODO: subpass - external
			rp.rpci.dependencyCount = deps.size();
			rp.rpci.pDependencies = deps.data();

			// attachments
			auto& ivs = rp.fbci.attachments;
			for (auto& attrpinfo : rp.attachments) {
				rp.rpci.attachments.push_back(attrpinfo.description);
				ivs.push_back(attrpinfo.iv);
			}
			rp.rpci.attachmentCount = rp.rpci.attachments.size();
			rp.rpci.pAttachments = rp.rpci.attachments.data();

			rp.handle = ptc.renderpass_cache.acquire(rp.rpci);
			rp.fbci.renderPass = rp.handle;
			rp.fbci.width = rp.attachments[0].extents.width;
			rp.fbci.height = rp.attachments[0].extents.height;
			rp.fbci.pAttachments = &ivs[0];
			rp.fbci.attachmentCount = ivs.size();
			rp.fbci.layers = 1;
			rp.framebuffer = ptc.framebuffer_cache.acquire(rp.fbci);
		}
	}

	vk::CommandBuffer RenderGraph::execute(vuk::PerThreadContext& ptc) {
		auto cbufs = ptc.commandbuffer_pool.acquire(1);
		auto& cbuf = cbufs[0];

		vk::CommandBufferBeginInfo cbi;
		cbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		cbuf.begin(cbi);

		CommandBuffer cobuf(ptc, cbuf);
		for (auto& rpass : rpis) {
			vk::RenderPassBeginInfo rbi;
			rbi.renderPass = rpass.handle;
			rbi.framebuffer = rpass.framebuffer;
			rbi.renderArea = vk::Rect2D(vk::Offset2D{}, vk::Extent2D{rpass.fbci.width, rpass.fbci.height});
			vk::ClearColorValue ccv;
			ccv.setFloat32({ 0.3f, 0.3f, 0.3f, 1.f });
			vk::ClearValue cv;
			cv.setColor(ccv);
			vk::ClearDepthStencilValue cdv;
			cdv.depth = 1.f;
			cdv.stencil = 0;
			vk::ClearValue cv2;
			cv2.setDepthStencil(cdv);
			std::vector<vk::ClearValue> clears;
			clears.push_back(cv);
			clears.push_back(cv2);
			rbi.pClearValues = clears.data();
			rbi.clearValueCount = clears.size(); // TODO
			cbuf.beginRenderPass(rbi, vk::SubpassContents::eInline);
			for (size_t i = 0; i < rpass.subpasses.size(); i++) {
				auto& sp = rpass.subpasses[i];
				cobuf.ongoing_renderpass = std::pair<decltype(rpass), unsigned>( rpass, i );
				sp.pass->pass.execute(cobuf);
				if (i < rpass.subpasses.size() - 1)
					cbuf.nextSubpass(vk::SubpassContents::eInline);
			}
			cbuf.endRenderPass();
		}
		cbuf.end();
		return cbuf;
	}

	void RenderGraph::generate_graph_visualization() {
		std::cout << "digraph RG {\n";
		for (auto& p : passes) {
			std::cout << p.pass.name << "[shape = Mrecord, label = \"" << p.pass.executes_on << "| {{";
			for (auto& i : p.pass.read_attachments) {
				std::cout << "<" << name_to_node(i.name) << ">" << i.name << "|";
			}
			for (auto& i : p.pass.read_buffers) {
				std::cout << "<" << name_to_node(i.name) << ">" << i.name << "|";
			}
			std::cout << "\b ";
			std::cout << "} | { \\n" << p.pass.name << " } | {";
			for (auto& i : p.pass.write_attachments) {
				std::cout << "<" << name_to_node(i.name) << ">" << i.name << "|";
			}
			for (auto& i : p.pass.write_buffers) {
				std::cout << "<" << name_to_node(i.name) << ">" << i.name << "|";
			}
			std::cout << "\b ";
			std::cout << "}}" << "\"];\n";
		}
		for (auto& gi : global_inputs) {
			std::visit(overloaded{
			[&](const ::Buffer& th) {
				std::cout << name_to_node(th.name) << "[shape = invtriangle, label =\"" << th.name << "\"];\n";
				for (auto& p : passes) {
					if (contains(p.pass.read_buffers, th))
						std::cout << name_to_node(th.name) << " -> " << p.pass.name << ":" << name_to_node(th.name) << ";\n";
				}
			},
			[&](const Attachment& th) {
				std::cout << name_to_node(th.name) << "[shape = invtrapezium, label =\"" << th.name << "\"];\n";
				for (auto& p : passes) {
					if (contains(p.pass.read_attachments, th))
						std::cout << name_to_node(th.name) << " -> " << p.pass.name << ":" << name_to_node(th.name) << ";\n";
				}
			}
				}, gi);
		}

		for (auto& gi : global_outputs) {
			std::visit(overloaded{
			[&](const ::Buffer& th) {
				std::cout << name_to_node(th.name) << "[shape = triangle, label =\"" << th.name << "\"];\n";
				for (auto& p : passes) {
					if (contains(p.pass.write_buffers, th))
						std::cout << p.pass.name << ":" << name_to_node(th.name) << " -> " << name_to_node(th.name) << ";\n";
				}

			},
			[&](const Attachment& th) {
				std::cout << name_to_node(th.name) << "[shape = trapezium, label =\"" << th.name << "\"];\n";
				for (auto& p : passes) {
					if (contains(p.pass.write_attachments, th))
						std::cout << p.pass.name << ":" << name_to_node(th.name) << " -> " << name_to_node(th.name) << ";\n";
				}

			}
				}, gi);
		}
		for (auto& gi : tracked) {
			std::visit(overloaded{
				// write_buffers -> sync -> sync -> read_buffers
				[&](const ::Buffer& th) {
					std::string src_node;
					Sync sync_out;
					std::vector<std::pair<std::string, Sync>> dst_nodes;
					for (auto& p : passes) {
						if (contains(p.pass.write_buffers, th)) {
							src_node = std::string(p.pass.name);
							sync_out = p.sync_out;
						}
						if (contains(p.pass.read_buffers, th)) {
							dst_nodes.emplace_back(std::string(p.pass.name), p.sync_in);
						}
					}
					std::string current_node = name_to_node(src_node) + ":" + name_to_node(th.name);
					if (auto xfer = contains_if(sync_out.queue_transfers, [&](auto& xfer) { return xfer.buffer == th.name; })) {
						std::cout << name_to_node(src_node) << "_" << name_to_node(th.name) << "_xfer_out" << "[label =\"" << xfer->queue_src << "->" << xfer->queue_dst << "\"];\n";
						std::cout << name_to_node(src_node) << ":" << name_to_node(th.name) << " -> " << name_to_node(src_node) << "_" << name_to_node(th.name) << "_xfer_out;\n";
						current_node = name_to_node(src_node) + "_" + name_to_node(th.name) + "_xfer_out";
					}

					for (auto& mlt : dst_nodes) {
						std::cout << current_node << "->" << mlt.first << ":" << name_to_node(th.name) << ";\n";
					}
				},
				[&](const Attachment& th) {
					std::vector<std::string> src_nodes, dst_nodes;
					for (auto& p : passes) {
						if (contains(p.pass.write_attachments, th)) {
							src_nodes.push_back(std::string(p.pass.name));
						}
						if (contains(p.pass.read_attachments, th)) {
							dst_nodes.push_back(std::string(p.pass.name));
						}
					}
					auto& multip = src_nodes.size() > 1 ? src_nodes : dst_nodes;
					auto& single = src_nodes.size() > 1 ? dst_nodes[0] : src_nodes[0];
					for (auto& mlt : multip) {
						std::cout << name_to_node(single) << ":" << name_to_node(th.name) << " -> " << mlt << ":" << name_to_node(th.name) << ";\n";
					}
				}
				}, gi);
		}


		std::cout << "}\n";
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, vk::Viewport vp) {
		command_buffer.setViewport(index, vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, Area area) {
		vk::Viewport vp;
		vp.x = area.offset.x;
		vp.y = area.offset.y;
		vp.width = area.extent.width;
		vp.height = area.extent.height;
		vp.minDepth = 0.f;
		vp.maxDepth = 1.f;
		command_buffer.setViewport(index, vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_viewport(unsigned index, Area::Framebuffer area) {
		assert(ongoing_renderpass);
		auto fb_dimensions = vk::Extent2D{ ongoing_renderpass->first.fbci.width, ongoing_renderpass->first.fbci.height };
		vk::Viewport vp;
		vp.x = area.x * fb_dimensions.width;
		vp.height = -area.height * fb_dimensions.height;
		vp.y = area.y * fb_dimensions.height - vp.height;
		vp.width = area.width * fb_dimensions.width;
		vp.minDepth = 0.f;
		vp.maxDepth = 1.f;
		command_buffer.setViewport(index, vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, vk::Rect2D vp) {
		command_buffer.setScissor(index, vp);
		return *this;
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, Area area) {
		command_buffer.setScissor(index, vk::Rect2D{area.offset, area.extent});
		return *this;
	}

	CommandBuffer& CommandBuffer::set_scissor(unsigned index, Area::Framebuffer area) {
		assert(ongoing_renderpass);
		auto fb_dimensions = vk::Extent2D{ ongoing_renderpass->first.fbci.width, ongoing_renderpass->first.fbci.height };
		vk::Rect2D vp;
		vp.offset.x = area.x * fb_dimensions.width;
		vp.offset.y = area.y * fb_dimensions.height;
		vp.extent.width = area.width * fb_dimensions.width;
		vp.extent.height = area.height * fb_dimensions.height;
		command_buffer.setScissor(index, vp);
		return *this;
	}


	CommandBuffer& CommandBuffer::bind_pipeline(vuk::PipelineCreateInfo pi) {
		pi.gpci.renderPass = ongoing_renderpass->first.handle;
		pi.gpci.subpass = ongoing_renderpass->second;
		current_pipeline = ptc.pipeline_cache.acquire(pi);
		command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, current_pipeline->pipeline);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_pipeline(Name p) {
		return bind_pipeline(ptc.ifc.ctx.named_pipelines.at(p));
	}

	CommandBuffer& CommandBuffer::bind_vertex_buffer(Allocator::Buffer& buf) {
		command_buffer.bindVertexBuffers(0, buf.buffer, buf.offset);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_index_buffer(Allocator::Buffer& buf, vk::IndexType type) {
		command_buffer.bindIndexBuffer(buf.buffer, buf.offset, type);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_sampled_image(unsigned set, unsigned binding, vk::ImageView iv, vk::SamplerCreateInfo sci) {
		sets_used[set] = true;
		set_bindings[set].bindings[binding].type = vk::DescriptorType::eCombinedImageSampler;
		set_bindings[set].bindings[binding].image = { };
		set_bindings[set].bindings[binding].image.imageView = iv;
		set_bindings[set].bindings[binding].image.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
		set_bindings[set].bindings[binding].image.sampler = ptc.sampler_cache.acquire(sci);
		set_bindings[set].used.set(binding);

		return *this;
	}

	CommandBuffer& CommandBuffer::push_constants(vk::ShaderStageFlags stages, size_t offset, void* data, size_t size) {
		assert(current_pipeline);
		command_buffer.pushConstants(current_pipeline->pipeline_layout, stages, offset, size, data);
		return *this;
	}

	CommandBuffer& CommandBuffer::bind_uniform_buffer(unsigned set, unsigned binding, Allocator::Buffer buffer) {
		sets_used[set] = true;
		set_bindings[set].bindings[binding].type = vk::DescriptorType::eUniformBuffer;
		set_bindings[set].bindings[binding].buffer = vk::DescriptorBufferInfo{ buffer.buffer, buffer.offset, buffer.size };
		set_bindings[set].used.set(binding);
		return *this;
	}

	CommandBuffer& CommandBuffer::draw(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
		_bind_graphics_pipeline_state();
		/* execute command */
		command_buffer.draw(a, b, c, d);
		return *this;
	}

	CommandBuffer& CommandBuffer::draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance) {
		_bind_graphics_pipeline_state();
		/* execute command */
		command_buffer.drawIndexed(index_count, instance_count, first_index, vertex_offset, first_instance);
		return *this;
	}

	void CommandBuffer::_bind_graphics_pipeline_state() {
		assert(current_pipeline);
		for (size_t i = 0; i < VUK_MAX_SETS; i++) {
			if (!sets_used[i])
				continue;
			set_bindings[i].layout_info = current_pipeline->layout_info;
			auto ds = ptc.descriptor_sets.acquire(set_bindings[i]);
			command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, current_pipeline->pipeline_layout, 0, 1, &ds, 0, nullptr);
			sets_used[i] = false;
			set_bindings[i] = {};
		}
	}
}
