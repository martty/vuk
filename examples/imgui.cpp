#include <stdint.h>

#include "imgui_frag.hpp"
#include "imgui_vert.hpp"
#include "utils.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/CommandBuffer.hpp"
#include "vuk/Context.hpp"
#include "vuk/Partials.hpp"
#include "vuk/RenderGraph.hpp"
#include "vuk/SampledImage.hpp"

using namespace vuk;

util::ImGuiData util::ImGui_ImplVuk_Init(Allocator& allocator) {
	Context& ctx = allocator.get_context();
	auto& io = ImGui::GetIO();
	io.BackendRendererName = "imgui_impl_vuk";
	io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset; // We can honor the ImDrawCmd::VtxOffset field, allowing for large meshes.

	unsigned char* pixels;
	int width, height;
	io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

	ImGuiData data;
	auto font_ia =
	    ImageAttachment::from_preset(ImageAttachment::Preset::eMap2D, Format::eR8G8B8A8Srgb, Extent3D{ (unsigned)width, (unsigned)height, 1u }, Samples::e1);
	auto [image, view, fut] = create_image_and_view_with_data(allocator, DomainFlagBits::eTransferOnTransfer, font_ia, pixels);
	data.font_image = std::move(image);
	data.font_image_view = std::move(view);
	Compiler comp;
	fut.wait(allocator, comp);
	ctx.set_name(data.font_image_view->payload, "ImGui/font");
	SamplerCreateInfo sci;
	sci.minFilter = sci.magFilter = Filter::eLinear;
	sci.mipmapMode = SamplerMipmapMode::eLinear;
	sci.addressModeU = sci.addressModeV = sci.addressModeW = SamplerAddressMode::eRepeat;
	data.font_sci = sci;
	data.font_si = std::make_unique<SampledImage>(SampledImage::Global{ *data.font_image_view, sci, ImageLayout::eReadOnlyOptimalKHR });
	io.Fonts->TexID = (ImTextureID)data.font_si.get();
	{
		PipelineBaseCreateInfo pci;
		// glslangValidator.exe -V imgui.vert --vn imgui_vert -o examples/imgui_vert.hpp
		pci.add_static_spirv(imgui_vert, sizeof(imgui_vert) / 4, "imgui.vert");
		// glslangValidator.exe -V imgui.frag --vn imgui_frag -o examples/imgui_frag.hpp
		pci.add_static_spirv(imgui_frag, sizeof(imgui_frag) / 4, "imgui.frag");
		ctx.create_named_pipeline("imgui", pci);
	}
	return data;
}

Value<ImageAttachment> util::ImGui_ImplVuk_Render(Allocator& allocator,
                                                  Value<ImageAttachment> target,
                                                  util::ImGuiData& data,
                                                  ImDrawData* draw_data,
                                                  const std::vector<Value<ImageAttachment>>& sampled_images) {
	auto reset_render_state = [](const util::ImGuiData& data, CommandBuffer& command_buffer, ImDrawData* draw_data, Buffer vertex, Buffer index) {
		command_buffer.bind_image(0, 0, *data.font_image_view).bind_sampler(0, 0, data.font_sci);
		if (index.size > 0) {
			command_buffer.bind_index_buffer(index, sizeof(ImDrawIdx) == 2 ? IndexType::eUint16 : IndexType::eUint32);
		}
		command_buffer.bind_vertex_buffer(0, vertex, 0, Packed{ Format::eR32G32Sfloat, Format::eR32G32Sfloat, Format::eR8G8B8A8Unorm });
		command_buffer.bind_graphics_pipeline("imgui");
		command_buffer.set_viewport(0, Rect2D::framebuffer());
		struct PC {
			float scale[2];
			float translate[2];
		} pc;
		pc.scale[0] = 2.0f / draw_data->DisplaySize.x;
		pc.scale[1] = 2.0f / draw_data->DisplaySize.y;
		pc.translate[0] = -1.0f - draw_data->DisplayPos.x * pc.scale[0];
		pc.translate[1] = -1.0f - draw_data->DisplayPos.y * pc.scale[1];
		command_buffer.push_constants(ShaderStageFlagBits::eVertex, 0, pc);
	};

	size_t vertex_size = draw_data->TotalVtxCount * sizeof(ImDrawVert);
	size_t index_size = draw_data->TotalIdxCount * sizeof(ImDrawIdx);
	auto imvert = *allocate_buffer(allocator, { MemoryUsage::eCPUtoGPU, vertex_size, 1 });
	auto imind = *allocate_buffer(allocator, { MemoryUsage::eCPUtoGPU, index_size, 1 });

	size_t vtx_dst = 0, idx_dst = 0;
	Compiler comp;
	for (int n = 0; n < draw_data->CmdListsCount; n++) {
		const ImDrawList* cmd_list = draw_data->CmdLists[n];
		auto imverto = imvert->add_offset(vtx_dst * sizeof(ImDrawVert));
		auto imindo = imind->add_offset(idx_dst * sizeof(ImDrawIdx));

		// TODO:
		host_data_to_buffer(allocator, DomainFlagBits{}, imverto, std::span(cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size)).wait(allocator, comp);
		host_data_to_buffer(allocator, DomainFlagBits{}, imindo, std::span(cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size)).wait(allocator, comp);
		vtx_dst += cmd_list->VtxBuffer.Size;
		idx_dst += cmd_list->IdxBuffer.Size;
	}

	// add rendergraph dependencies to be transitioned
	// make all rendergraph sampled images available
	auto sampled_images_array = declare_array("imgui_sampled", std::span(sampled_images));

	auto pass = make_pass("imgui",
	                      [&data, &allocator, verts = imvert.get(), inds = imind.get(), draw_data, reset_render_state](
	                          CommandBuffer& command_buffer, VUK_IA(Access::eColorWrite) dst, VUK_ARG(ImageAttachment[], Access::eFragmentSampled) sis) {
		                      command_buffer.set_dynamic_state(DynamicStateFlagBits::eViewport | DynamicStateFlagBits::eScissor);
		                      command_buffer.set_rasterization(PipelineRasterizationStateCreateInfo{});
		                      command_buffer.set_color_blend(dst, BlendPreset::eAlphaBlend);
		                      reset_render_state(data, command_buffer, draw_data, verts, inds);
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
						                      reset_render_state(data, command_buffer, draw_data, verts, inds);
					                      else
						                      pcmd->UserCallback(cmd_list, pcmd);
				                      } else {
					                      // Project scissor/clipping rectangles into framebuffer space
					                      ImVec4 clip_rect;
					                      clip_rect.x = (pcmd->ClipRect.x - clip_off.x) * clip_scale.x;
					                      clip_rect.y = (pcmd->ClipRect.y - clip_off.y) * clip_scale.y;
					                      clip_rect.z = (pcmd->ClipRect.z - clip_off.x) * clip_scale.x;
					                      clip_rect.w = (pcmd->ClipRect.w - clip_off.y) * clip_scale.y;

					                      auto fb_width = command_buffer.get_ongoing_render_pass().extent.width;
					                      auto fb_height = command_buffer.get_ongoing_render_pass().extent.height;
					                      if (clip_rect.x < fb_width && clip_rect.y < fb_height && clip_rect.z >= 0.0f && clip_rect.w >= 0.0f) {
						                      // Negative offsets are illegal for vkCmdSetScissor
						                      if (clip_rect.x < 0.0f)
							                      clip_rect.x = 0.0f;
						                      if (clip_rect.y < 0.0f)
							                      clip_rect.y = 0.0f;

						                      // Apply scissor/clipping rectangle
						                      Rect2D scissor;
						                      scissor.offset.x = (int32_t)(clip_rect.x);
						                      scissor.offset.y = (int32_t)(clip_rect.y);
						                      scissor.extent.width = (uint32_t)(clip_rect.z - clip_rect.x);
						                      scissor.extent.height = (uint32_t)(clip_rect.w - clip_rect.y);
						                      command_buffer.set_scissor(0, scissor);

						                      // Bind texture
						                      if (pcmd->TextureId) {
							                      auto si_index = reinterpret_cast<uint64_t>(pcmd->TextureId);

							                      command_buffer.bind_image(0, 0, sis[si_index]).bind_sampler(0, 0, {});

							                      // TODO: SampledImage
							                      //.bind_sampler(0, 0, sis[si_index]);
						                      }
						                      // Draw
						                      command_buffer.draw_indexed(pcmd->ElemCount, 1, pcmd->IdxOffset + global_idx_offset, pcmd->VtxOffset + global_vtx_offset, 0);
					                      }
				                      }
			                      }
			                      global_idx_offset += cmd_list->IdxBuffer.Size;
			                      global_vtx_offset += cmd_list->VtxBuffer.Size;
		                      }
		                      return dst;
	                      });

	return pass(target, sampled_images_array);
}
