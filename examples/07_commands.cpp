#include "example_runner.hpp"
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <numeric>
#include <random>
#include <stb_image.h>

/* 07_commands
 * In this example we will see how to create passes that execute outside of renderpasses.
 * To showcase this, we will manually resolve an MSAA image (from the previous example),
 * then blit parts of it to the final image.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	float time = 0.f;
	bool start = false;
	auto box = util::generate_cube();
	std::optional<vuk::Texture> texture_of_doge;
	std::vector<unsigned> shuf(9);

	vuk::Example x{
		.name = "07_commands",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      // Same setup as for 04_texture
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file("../../examples/ubo_test_tex.vert"), "ubo_test_tex.vert");
			      pci.add_glsl(util::read_entire_file("../../examples/triangle_depthshaded_tex.frag"), "triangle_depthshaded_text.frag");
			      runner.context->create_named_pipeline("textured_cube", pci);
		      }

		      int x, y, chans;
		      auto doge_image = stbi_load("../../examples/doge.png", &x, &y, &chans, 4);

		      auto [tex, tex_fut] = create_texture(allocator, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, doge_image, false);
		      texture_of_doge = std::move(tex);
		      runner.enqueue_setup(std::move(tex_fut));

		      // Init tiles
		      std::iota(shuf.begin(), shuf.end(), 0);
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      // We set up the cube data, same as in example 02_cube
		      auto [vert_buf, vert_fut] = create_buffer_gpu(frame_allocator, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.first));
		      auto verts = *vert_buf;
		      auto [ind_buf, ind_fut] = create_buffer_gpu(frame_allocator, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.second));
		      auto inds = *ind_buf;

		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      vp.view = glm::lookAt(glm::vec3(0, 0, 1.75), glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 0.1f, 10.f);
		      vp.proj[1][1] *= -1;

		      auto [buboVP, uboVP_fut] = create_buffer_cross_device(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      vuk::wait_for_futures(frame_allocator, vert_fut, ind_fut, uboVP_fut);

		      vuk::RenderGraph rg;

		      // The rendering pass is unchanged by going to multisampled,
		      // but we will use an offscreen multisampled color attachment
		      rg.add_pass({ .resources = { "07_commands_MS"_image >> vuk::eColorWrite, "07_commands_depth"_image >> vuk::eDepthStencilRW },
		                    .execute = [verts, uboVP, inds](vuk::CommandBuffer& command_buffer) {
			                    command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			                        .set_scissor(0, vuk::Rect2D::framebuffer())
			                        .set_rasterization({}) // Set the default rasterization state
			                        // Set the depth/stencil state
			                        .set_depth_stencil(vuk::PipelineDepthStencilStateCreateInfo{
			                            .depthTestEnable = true,
			                            .depthWriteEnable = true,
			                            .depthCompareOp = vuk::CompareOp::eLessOrEqual,
			                        })
			                        .broadcast_color_blend({}) // Set the default color blend state
			                        .bind_vertex_buffer(0,
			                                            verts,
			                                            0,
			                                            vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
			                                                         vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - sizeof(util::Vertex::position) },
			                                                         vuk::Format::eR32G32Sfloat })
			                        .bind_index_buffer(inds, vuk::IndexType::eUint32)
			                        .bind_image(0, 2, *texture_of_doge->view)
			                        .bind_sampler(0, 2, {})
			                        .bind_graphics_pipeline("textured_cube")
			                        .bind_buffer(0, 0, uboVP);
			                    glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
			                    *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(0.f), glm::vec3(0.f, 1.f, 0.f)));
			                    command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
		                    } });
		      // Add a pass where we resolve our multisampled image
		      // Since we didn't declare any framebuffer forming resources, this pass will execute outside of a renderpass
		      // Hence we can only call commands that are valid outside of a renderpass
		      rg.add_pass({ .name = "07_resolve",
		                    .resources = { "07_commands_MS+"_image >> vuk::eTransferRead, "07_commands_NMS"_image >> vuk::eTransferWrite },
		                    .execute = [](vuk::CommandBuffer& command_buffer) {
			                    command_buffer.resolve_image("07_commands_MS+", "07_commands_NMS");
		                    } });

		      float tile_x_size = 300 / 3;
		      float tile_y_size = 300 / 3;

		      // Here we demonstrate blitting by splitting up the resolved image into 09 tiles
		      // And blitting those tiles in the order dictated by 'shuf'
		      // We will also sort shuf over time, to show a nice animation
		      rg.add_pass({ .name = "07_blit",
		                    .resources = { "07_commands_NMS+"_image >> vuk::eTransferRead, "07_commands"_image >> vuk::eTransferWrite >> "07_commands_final" },
		                    .execute = [tile_x_size, tile_y_size](vuk::CommandBuffer& command_buffer) {
			                    for (auto i = 0; i < 9; i++) {
				                    auto x = i % 3;
				                    auto y = i / 3;

				                    auto sx = shuf[i] % 3;
				                    auto sy = shuf[i] / 3;

				                    vuk::ImageBlit blit;
				                    blit.srcSubresource.aspectMask = vuk::ImageAspectFlagBits::eColor;
				                    blit.srcSubresource.baseArrayLayer = 0;
				                    blit.srcSubresource.layerCount = 1;
				                    blit.srcSubresource.mipLevel = 0;
				                    blit.srcOffsets[0] = vuk::Offset3D{ static_cast<int>(x * tile_x_size), static_cast<int>(y * tile_y_size), 0 };
				                    blit.srcOffsets[1] = vuk::Offset3D{ static_cast<int>((x + 1) * tile_x_size), static_cast<int>((y + 1) * tile_y_size), 1 };
				                    blit.dstSubresource = blit.srcSubresource;
				                    blit.dstOffsets[0] = vuk::Offset3D{ static_cast<int>(sx * tile_x_size), static_cast<int>(sy * tile_y_size), 0 };
				                    blit.dstOffsets[1] = vuk::Offset3D{ static_cast<int>((sx + 1) * tile_x_size), static_cast<int>((sy + 1) * tile_y_size), 1 };
				                    command_buffer.blit_image("07_commands_NMS+", "07_commands_final", blit, vuk::Filter::eLinear);
			                    }
		                    } });

		      time += ImGui::GetIO().DeltaTime;
		      if (!start && time > 5.f) {
			      start = true;
			      time = 0;
			      std::random_device rd;
			      std::mt19937 g(rd());
			      std::shuffle(shuf.begin(), shuf.end(), g);
		      }
		      if (start && time > 1.f) {
			      time = 0;
			      // World's slowest bubble sort, one iteration per second
			      bool swapped = false;
			      for (unsigned i = 1; i < shuf.size(); i++) {
				      if (shuf[i - 1] > shuf[i]) {
					      std::swap(shuf[i - 1], shuf[i]);
					      swapped = true;
					      break;
				      }
			      }
			      // 'shuf' is sorted, restart
			      if (!swapped) {
				      start = false;
			      }
		      }

		      // We mark our MS attachment as multisampled (8 samples)
		      // from the final image, and we don't need to specify here
		      // We use the swapchain format & extents, since resolving needs identical formats & extents
		      rg.attach_managed(
		          "07_commands_MS", runner.swapchain->format, vuk::Dimension2D::absolute(300, 300), vuk::Samples::e8, vuk::ClearColor{ 0.f, 0.f, 0.f, 0.f });
		      rg.attach_managed(
		          "07_commands_depth", vuk::Format::eD32Sfloat, vuk::Dimension2D::framebuffer(), vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });
		      rg.attach_managed(
		          "07_commands_NMS", runner.swapchain->format, vuk::Dimension2D::absolute(300, 300), vuk::Samples::e1, vuk::ClearColor{ 0.f, 0.f, 0.f, 0.f });

		      return vuk::Future<vuk::ImageAttachment>{ frame_allocator, std::make_unique<vuk::RenderGraph>(std::move(rg)), "07_commands_final" };
		    },
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      texture_of_doge.reset();
		    }

	};

	REGISTER_EXAMPLE(x);
} // namespace