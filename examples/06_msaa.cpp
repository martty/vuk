#include "example_runner.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <stb_image.h>

/* 06_msaa
 * In this example we will build on the previous example 04_texture, but we will render the cube to a multisampled texture,
 * which we will resolve to the final swapchain image.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();
	vuk::Unique<vuk::Buffer> verts, inds;
	vuk::Unique<vuk::Image> image_of_doge;
	vuk::Unique<vuk::ImageView> image_view_of_doge;
	vuk::ImageAttachment texture_of_doge;

	vuk::Example x{
		.name = "06_msaa",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      // Same setup as for 04_texture, except we use spirv to create the pipeline
		      // This is a good option is if you don't want to ship shaderc or if you are caching or have your sl -> spirv pipeline
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_spirv(util::read_spirv((root / "examples/ubo_test_tex.vert.spv").generic_string()), (root / "examples/ubo_test_tex.vert").generic_string());
			      pci.add_spirv(util::read_spirv((root / "examples/triangle_depthshaded_tex.frag.spv").generic_string()),
			                    (root / "examples/triangle_depthshaded_text.frag").generic_string());
			      runner.context->create_named_pipeline("textured_cube", pci);
		      }

		      int x, y, chans;
		      auto doge_image = stbi_load((root / "examples/doge.png").generic_string().c_str(), &x, &y, &chans, 4);

		      texture_of_doge = vuk::ImageAttachment::from_preset(
		          vuk::ImageAttachment::Preset::eMap2D, vuk::Format::eR8G8B8A8Srgb, vuk::Extent3D{ (unsigned)x, (unsigned)y, 1u }, vuk::Samples::e1);
		      texture_of_doge.level_count = 1;
		      auto [image, view, future] = vuk::create_image_and_view_with_data(allocator, vuk::DomainFlagBits::eTransferOnTransfer, texture_of_doge, doge_image);
		      image_of_doge = std::move(image);
		      image_view_of_doge = std::move(view);
		      runner.enqueue_setup(future.as_released(vuk::Access::eFragmentSampled, vuk::DomainFlagBits::eGraphicsQueue));
		      stbi_image_free(doge_image);

		      // We set up the cube data, same as in example 02_cube
		      auto [vert_buf, vert_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.first));
		      verts = std::move(vert_buf);
		      auto [ind_buf, ind_fut] = create_buffer(allocator, vuk::MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnGraphics, std::span(box.second));
		      inds = std::move(ind_buf);
		      // For the example, we just ask these that these uploads complete before moving on to rendering
		      // In an engine, you would integrate these uploads into some explicit system
		      runner.enqueue_setup(std::move(vert_fut));
		      runner.enqueue_setup(std::move(ind_fut));
		    },
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Value<vuk::ImageAttachment> target) {
		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      vp.view = glm::lookAt(glm::vec3(0, 1.5, 3.5), glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
		      vp.proj[1][1] *= -1;

		      auto [buboVP, uboVP_fut] = create_buffer(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      // The rendering pass is unchanged by going to multisampled,
		      // but we will use an offscreen multisampled color attachment
		      auto ms_render =
		          vuk::make_pass("06_msaa_render", [uboVP](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eColorWrite) color, VUK_IA(vuk::eDepthStencilRW) depth) {
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
			                                  *verts,
			                                  0,
			                                  vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
			                                               vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - sizeof(util::Vertex::position) },
			                                               vuk::Format::eR32G32Sfloat })
			              .bind_index_buffer(*inds, vuk::IndexType::eUint32)
			              .bind_image(0, 2, *image_view_of_doge)
			              .bind_sampler(0, 2, {})
			              .bind_graphics_pipeline("textured_cube")
			              .bind_buffer(0, 0, uboVP);
			          glm::mat4* model = command_buffer.scratch_buffer<glm::mat4>(0, 1);
			          *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
			          command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);

					  return color;
		          });
		      angle += 180.f * ImGui::GetIO().DeltaTime;

		      // We mark our MS attachment as multisampled (8 samples)
		      // Since resolving requires equal sized images, we can actually infer the size of the MS attachment
		      // from the final image, and we don't need to specify here
		      // We use the swapchain format, since resolving needs identical formats
		      auto ms_img = vuk::declare_ia("06_ms");
		      ms_img->sample_count = vuk::Samples::e8;
		      ms_img = vuk::clear_image(std::move(ms_img), vuk::ClearColor{ 0.f, 0.f, 0.f, 0.f });
		      
			  auto depth_img = vuk::declare_ia("06_depth");
		      depth_img->format = vuk::Format::eD32Sfloat;
		      depth_img = vuk::clear_image(std::move(depth_img), vuk::ClearDepthStencil{ 1.0f, 0 });

		      // We mark our final result "06_msaa_final" attachment to be a result of a resolve from "06_msaa_MS"
		      return vuk::resolve_into(ms_render(std::move(ms_img), std::move(depth_img)), std::move(target));
		    },
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      verts.reset();
		      inds.reset();
		      image_of_doge.reset();
		      image_view_of_doge.reset();
		    }

	};

	REGISTER_EXAMPLE(x);
} // namespace