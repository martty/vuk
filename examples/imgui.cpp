#include "utils.hpp"
#include "Context.hpp"
#include "CommandBuffer.hpp"
#include "RenderGraph.hpp"

util::ImGuiData util::ImGui_ImplVuk_Init(vuk::PerThreadContext& ptc) {
	auto& io = ImGui::GetIO();
	io.BackendRendererName = "imgui_impl_vuk";
	io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;  // We can honor the ImDrawCmd::VtxOffset field, allowing for large meshes.

	unsigned char* pixels;
	int width, height;
	io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

	ImGuiData data;
	auto [tex, stub] = ptc.create_texture(vk::Format::eR8G8B8A8Srgb, vk::Extent3D(width, height, 1), pixels);
	data.font_texture = std::move(tex);
	ptc.ctx.debug.set_name(data.font_texture, "ImGui/font");
	vk::SamplerCreateInfo sci;
	sci.minFilter = sci.magFilter = vk::Filter::eLinear;
	sci.mipmapMode = vk::SamplerMipmapMode::eLinear;
	sci.addressModeU = sci.addressModeV = sci.addressModeW = vk::SamplerAddressMode::eRepeat;
	sci.minLod = -1000;
	sci.maxLod = 1000;
	sci.maxAnisotropy = 1.0f;
	data.font_sci = sci;
	data.font_si = std::make_unique<vuk::SampledImage>(vuk::SampledImage::Global{ *data.font_texture.view, sci, vk::ImageLayout::eShaderReadOnlyOptimal });
	io.Fonts->TexID = (ImTextureID)data.font_si.get();
	{
		vuk::PipelineBaseCreateInfo pci;
		auto vpath = "../../imgui.vert";
		auto vcont = util::read_entire_file(vpath);
		pci.add_shader(vcont, vpath);
		auto fpath = "../../imgui.frag";
		auto fcont = util::read_entire_file(fpath);
		pci.add_shader(fcont, fpath);
		pci.set_blend(vuk::BlendPreset::eAlphaBlend);
		ptc.ctx.create_named_pipeline("imgui", pci);
	}

	return data;
}

vuk::Pass util::ImGui_ImplVuk_Render(vuk::PerThreadContext& ptc, vuk::Name src_target, vuk::Name dst_target, util::ImGuiData& data, ImDrawData* draw_data) {
	auto reset_render_state = [](const util::ImGuiData& data, vuk::CommandBuffer& command_buffer, ImDrawData* draw_data, vuk::Buffer vertex, vuk::Buffer index) {
		command_buffer.bind_sampled_image(0, 0, *data.font_texture.view, data.font_sci);
		if (index.size > 0) {
			command_buffer.bind_index_buffer(index, sizeof(ImDrawIdx) == 2 ? vk::IndexType::eUint16 : vk::IndexType::eUint32);
		}
		command_buffer.bind_vertex_buffer(0, vertex, 0, vuk::Packed{ vk::Format::eR32G32Sfloat, vk::Format::eR32G32Sfloat, vk::Format::eR8G8B8A8Unorm });
		command_buffer.bind_graphics_pipeline("imgui");
		command_buffer.set_viewport(0, vuk::Area::Framebuffer{});
		struct PC {
			float scale[2];
			float translate[2];
		} pc;
		pc.scale[0] = 2.0f / draw_data->DisplaySize.x;
		pc.scale[1] = -2.0f / draw_data->DisplaySize.y;
		pc.translate[0] = -1.0f - draw_data->DisplayPos.x * pc.scale[0];
		pc.translate[1] = 1.0f + draw_data->DisplayPos.y * pc.scale[1];
		command_buffer.push_constants(vk::ShaderStageFlagBits::eVertex, 0, pc);
	};

	size_t vertex_size = draw_data->TotalVtxCount * sizeof(ImDrawVert);
	size_t index_size = draw_data->TotalIdxCount * sizeof(ImDrawIdx);
	auto imvert = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst, vertex_size, false);
	auto imind = ptc._allocate_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, index_size, false);

	size_t vtx_dst = 0, idx_dst = 0;
	for (int n = 0; n < draw_data->CmdListsCount; n++) {
		const ImDrawList* cmd_list = draw_data->CmdLists[n];
		auto imverto = imvert;
		imverto.offset += vtx_dst * sizeof(ImDrawVert);
		auto imindo = imind;
		imindo.offset += idx_dst * sizeof(ImDrawIdx);

		ptc.upload(imverto, std::span(cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size));
		ptc.upload(imindo, std::span(cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size));
		vtx_dst += cmd_list->VtxBuffer.Size;
		idx_dst += cmd_list->IdxBuffer.Size;
	}

	ptc.wait_all_transfers();
	vuk::Pass pass{
		.resources = { vuk::Resource{src_target, dst_target, vuk::Resource::Type::eImage, vuk::eColorRW} },
		.execute = [&data, imvert, imind, draw_data, reset_render_state](vuk::CommandBuffer& command_buffer) {
			reset_render_state(data, command_buffer, draw_data, imvert, imind);
			// Will project scissor/clipping rectangles into framebuffer space
			ImVec2 clip_off = draw_data->DisplayPos;         // (0,0) unless using multi-viewports
			ImVec2 clip_scale = draw_data->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

			// Render command lists
			// (Because we merged all buffers into a single one, we maintain our own offset into them)
			int global_vtx_offset = 0;
			int global_idx_offset = 0;
			for (int n = 0; n < draw_data->CmdListsCount; n++) {
				const ImDrawList* cmd_list = draw_data->CmdLists[n];
				for (int cmd_i = 0; cmd_i < cmd_list->CmdBuffer.Size; cmd_i++) {
					const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[cmd_i];
					if (pcmd->UserCallback != nullptr) {
						// User callback, registered via ImDrawList::AddCallback()
						// (ImDrawCallback_ResetRenderState is a special callback value used by the user to request the renderer to reset render state.)
						if (pcmd->UserCallback == ImDrawCallback_ResetRenderState)
							reset_render_state(data, command_buffer, draw_data, imvert, imind);
						else
							pcmd->UserCallback(cmd_list, pcmd);
					} else {
						// Project scissor/clipping rectangles into framebuffer space
						ImVec4 clip_rect;
						clip_rect.x = (pcmd->ClipRect.x - clip_off.x) * clip_scale.x;
						clip_rect.y = (pcmd->ClipRect.y - clip_off.y) * clip_scale.y;
						clip_rect.z = (pcmd->ClipRect.z - clip_off.x) * clip_scale.x;
						clip_rect.w = (pcmd->ClipRect.w - clip_off.y) * clip_scale.y;

						auto fb_width = command_buffer.get_ongoing_renderpass().extent.width;
						auto fb_height = command_buffer.get_ongoing_renderpass().extent.height;
						if (clip_rect.x < fb_width && clip_rect.y < fb_height && clip_rect.z >= 0.0f && clip_rect.w >= 0.0f) {
							// Negative offsets are illegal for vkCmdSetScissor
							if (clip_rect.x < 0.0f)
								clip_rect.x = 0.0f;
							if (clip_rect.y < 0.0f)
								clip_rect.y = 0.0f;

							// Apply scissor/clipping rectangle
							VkRect2D scissor;
							scissor.offset.x = (int32_t)(clip_rect.x);
							scissor.offset.y = (int32_t)(clip_rect.y);
							scissor.extent.width = (uint32_t)(clip_rect.z - clip_rect.x);
							scissor.extent.height = (uint32_t)(clip_rect.w - clip_rect.y);
							command_buffer.set_scissor(0, scissor);

							// Bind texture
							if (pcmd->TextureId) {
								auto& si = *reinterpret_cast<vuk::SampledImage*>(pcmd->TextureId);
								if (si.is_global) {
									command_buffer.bind_sampled_image(0, 0, si.global.iv, si.global.sci);
								} else {
									if (si.rg_attachment.ivci) {
										command_buffer.bind_sampled_image(0, 0, si.rg_attachment.attachment_name, *si.rg_attachment.ivci, si.rg_attachment.sci);
									} else {
										command_buffer.bind_sampled_image(0, 0, si.rg_attachment.attachment_name, si.rg_attachment.sci);
									}
								}
							}
							// Draw
							command_buffer.draw_indexed(pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset, pcmd->VtxOffset + global_vtx_offset, 0);
						}
					}
				}
				global_idx_offset += cmd_list->IdxBuffer.Size;
				global_vtx_offset += cmd_list->VtxBuffer.Size;
			}
		}
	};

	// add rendergraph dependencies to be transitioned
	// make all rendergraph sampled images available
	for (auto& si : ptc.sampled_images.pool.values) {
		if (!si.is_global) {
			pass.resources.push_back(vuk::Resource(si.rg_attachment.attachment_name, vuk::Resource::Type::eImage, vuk::Access::eFragmentSampled));
		}
	}

	return pass;
}
