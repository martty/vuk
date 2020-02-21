#include "RenderGraph.hpp"
#include "Hash.hpp" // for create
#include "Context.hpp"

void RenderGraph::build(vuk::InflightContext& ifc) {
	// output attachments have an initial layout of eUndefined (loadOp: clear or dontcare, storeOp: store)
	for (auto& [name, attachment_info] : output_attachments) {
		// get the first pass this attachment is used
		auto pass = contains_if(passes, [&](auto& pi) {
			auto& p = pi.pass;
			if (contains_if(p.color_attachments, [&](auto& att) { return att.name == name; })) return true;
			return false;
			});
		// in the renderpass this pass participates in, we want to fill in the info for the attachment
		// TODO: we want to find the first to put in the initial layout
		// then we want to find the last to put in the final layout
		assert(pass);
		rpis[pass->render_pass_index].attachments[name] = attachment_info;
	}
	// do this for input attachments
	// input attachments have a finallayout matching last use
	// loadOp: load, storeOp: dontcare
	// inout attachments
	// loadOp: load, storeOp: store

	// -- transient attachments --

	// we now have enough data to build vk::RenderPasses and vk::Framebuffers
	for (auto& rp : rpis) {
		rp.rpci.attachmentCount = rp.attachments.size();
		std::vector<vk::AttachmentDescription> atts;
		for (auto& [name, attrpinfo] : rp.attachments) {
			atts.push_back(attrpinfo.description);
		}
		rp.rpci.pAttachments = atts.data();
		std::vector<vk::SubpassDescription> subp;
		for (auto& s : rp.subpasses) {
			vk::SubpassDescription sd;
			std::vector<vk::AttachmentReference>* attrefs = new std::vector<vk::AttachmentReference>;
			for (auto& [name, att] : s.attachments) {
				vk::AttachmentReference attref;
				attref.layout = att.layout;
				attref.attachment = 0; // TODO
				attrefs->push_back(attref);
			}
			sd.colorAttachmentCount = attrefs->size();
			sd.pColorAttachments = attrefs->data();
			subp.push_back(sd);
		}
		rp.rpci.pSubpasses = subp.data();
		rp.rpci.subpassCount = subp.size();

		rp.handle = ifc.renderpass_cache.acquire(rp.rpci);
		rp.fbci.attachmentCount = rp.attachments.size();
		rp.fbci.renderPass = rp.handle;
		rp.fbci.width = rp.attachments.begin()->second.extents.width;
		rp.fbci.height = rp.attachments.begin()->second.extents.height;
		rp.fbci.pAttachments = &rp.attachments.begin()->second.iv;
		rp.fbci.layers = 1;
		rp.framebuffer = ifc.ctx.device.createFramebuffer(rp.fbci);
	}
}

vk::CommandBuffer RenderGraph::execute(vuk::InflightContext& ifc) {
	auto pfc = ifc.begin();
	auto cbufs = pfc.commandbuffer_pool.acquire(1);
	auto& cbuf = cbufs[0];

	vk::CommandBufferBeginInfo cbi;
	cbi.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	cbuf.begin(cbi);

	CommandBuffer cobuf(ifc, cbuf);
	for (auto& rpass : rpis) {
		vk::RenderPassBeginInfo rbi;
		rbi.renderPass = rpass.handle;
		rbi.framebuffer = rpass.framebuffer;
		rbi.clearValueCount = 1; // TODO
		vk::ClearColorValue ccv;
		ccv.setFloat32({ 0.3f, 0.3f, 0.3f, 1.f });
		vk::ClearValue cv;
		cv.setColor(ccv);
		rbi.pClearValues = &cv;
		cbuf.beginRenderPass(rbi, vk::SubpassContents::eInline);
		size_t subpass_index = 0;
		for (auto& sp : rpass.subpasses) {
			cobuf.ongoing_renderpass = std::pair{rpass.handle, subpass_index};
			sp.pass->pass.execute(cobuf);
			//cbuf.nextSubpass(vk::SubpassContents::eInline);
			subpass_index++;
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
			[&](Buffer& th) {
				std::cout << name_to_node(th.name) << "[shape = invtriangle, label =\"" << th.name << "\"];\n";
				for (auto& p : passes) {
					if (contains(p.pass.read_buffers, th))
						std::cout << name_to_node(th.name) << " -> " << p.pass.name << ":" << name_to_node(th.name) << ";\n";
				}
			},
			[&](Attachment& th) {
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
			[&](Buffer& th) {
				std::cout << name_to_node(th.name) << "[shape = triangle, label =\"" << th.name << "\"];\n";
				for (auto& p : passes) {
					if (contains(p.pass.write_buffers, th))
						std::cout << p.pass.name << ":" << name_to_node(th.name) << " -> " << name_to_node(th.name) << ";\n";
				}

			},
			[&](Attachment& th) {
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
				[&](Buffer& th) {
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
				[&](Attachment& th) {
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

	CommandBuffer& CommandBuffer::draw(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
		/* flush graphics state */
		if (next_viewport) {
			command_buffer.setViewport(0, *next_viewport);
			next_viewport = {};
		}
		if (next_scissor) {
			command_buffer.setScissor(0, *next_scissor);
			next_scissor = {};
		}
		if (next_graphics_pipeline) {
			next_graphics_pipeline->renderPass = ongoing_renderpass->first;
			next_graphics_pipeline->subpass = ongoing_renderpass->second;
			command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, ifc.pipeline_cache.acquire(*next_graphics_pipeline));
			next_graphics_pipeline = {};
		}
		/* execute command */
		command_buffer.draw(a, b, c, d);
		return *this;
	}
