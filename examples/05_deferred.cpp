#include "example_runner.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>

/* 05_deferred
 * In this example we will take our cube to the next level by rendering it deferred.
 * To achieve this, we will first render the cube to three offscreen textures -
 * one containing the world position, the second the world normals and the third containing colour.
 * We will also have depth buffering for this draw.
 * After this, we will compute the shading by using a fullscreen pass, where we sample from these textures.
 * To achieve this, we will need to let the rendergraph know of our image dependencies.
 * Note that it is generally not a good idea to store position (since it can be reconstructed from depth).
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();
	vuk::Unique<vuk::Buffer> verts, inds;

	vuk::Example x{
		.name = "05_deferred",
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/deferred.vert").generic_string()), (root / "examples/deferred.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/deferred.frag").generic_string()), (root / "examples/deferred.frag").generic_string());
			      runner.context->create_named_pipeline("cube_deferred", pci);
		      }

		      {
			      vuk::PipelineBaseCreateInfo pci;
			      pci.add_glsl(util::read_entire_file((root / "examples/fullscreen.vert").generic_string()), (root / "examples/fullscreen.vert").generic_string());
			      pci.add_glsl(util::read_entire_file((root / "examples/deferred_resolve.frag").generic_string()),
			                   (root / "examples/deferred_resolve.frag").generic_string());
			      runner.context->create_named_pipeline("deferred_resolve", pci);
		      }

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
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Future target) {
		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      auto cam_pos = glm::vec3(0, 1.5, 3.5);
		      vp.view = glm::lookAt(cam_pos, glm::vec3(0), glm::vec3(0, 1, 0));
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
		      vp.proj[1][1] *= -1;

		      auto [buboVP, uboVP_fut] = create_buffer(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span(&vp, 1));
		      auto uboVP = *buboVP;

		      vuk::RenderGraph rg("05");
		      rg.attach_in("05_deferred", std::move(target));
		      // Here we will render the cube into 3 offscreen textures
		      rg.add_pass({ // Passes can be optionally named, this useful for visualization and debugging
		                    .name = "05_deferred_MRT",
		                    // Declare our framebuffer
		                    .resources = { "05_position"_image >> vuk::eColorWrite,
		                                   "05_normal"_image >> vuk::eColorWrite,
		                                   "05_color"_image >> vuk::eColorWrite,
		                                   "05_depth"_image >> vuk::eDepthStencilRW },
		                    .execute = [uboVP](vuk::CommandBuffer& command_buffer) {
			                    // Rendering is the same as in the case for forward
			                    command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			                        .set_scissor(0, vuk::Rect2D::framebuffer())
			                        .set_rasterization(vuk::PipelineRasterizationStateCreateInfo{}) // Set the default rasterization state
			                        // Set the depth/stencil state
			                        .set_depth_stencil(vuk::PipelineDepthStencilStateCreateInfo{
			                            .depthTestEnable = true,
			                            .depthWriteEnable = true,
			                            .depthCompareOp = vuk::CompareOp::eLessOrEqual,
			                        })
			                        .set_color_blend("05_position", {}) // Set the default color blend state individually for demonstration
			                        .set_color_blend("05_normal",
			                                         {}) // If you want to use different blending state per attachment, you must enable the independentBlend feature
			                        .set_color_blend("05_color", {})
			                        .bind_vertex_buffer(0,
			                                            *verts,
			                                            0,
			                                            vuk::Packed{ vuk::Format::eR32G32B32Sfloat,
			                                                         vuk::Format::eR32G32B32Sfloat,
			                                                         vuk::Ignore{ offsetof(util::Vertex, uv_coordinates) - offsetof(util::Vertex, tangent) },
			                                                         vuk::Format::eR32G32Sfloat })
			                        .bind_index_buffer(*inds, vuk::IndexType::eUint32)
			                        .bind_graphics_pipeline("cube_deferred")
			                        .bind_buffer(0, 0, uboVP);
			                    glm::mat4* model = command_buffer.map_scratch_buffer<glm::mat4>(0, 1);
			                    *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
			                    command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
		                    } });

		      angle += 360.f * ImGui::GetIO().DeltaTime;

		      // The shading pass for the deferred rendering
		      rg.add_pass({ .name = "05_deferred_resolve",
		                    // Declare that we are going to render to the final color image
		                    // Declare that we are going to sample (in the fragment shader) from the previous attachments
		                    .resources = { "05_deferred"_image >> vuk::eColorWrite >> "05_deferred_final",
		                                   "05_position+"_image >> vuk::eFragmentSampled,
		                                   "05_normal+"_image >> vuk::eFragmentSampled,
		                                   "05_color+"_image >> vuk::eFragmentSampled },
		                    .execute = [cam_pos](vuk::CommandBuffer& command_buffer) {
			                    command_buffer.set_viewport(0, vuk::Rect2D::framebuffer())
			                        .set_scissor(0, vuk::Rect2D::framebuffer())
			                        .set_rasterization({})     // Set the default rasterization state
			                        .broadcast_color_blend({}) // Set the default color blend state
			                        .bind_graphics_pipeline("deferred_resolve");
			                    // Set camera position so we can do lighting
			                    *command_buffer.map_scratch_buffer<glm::vec3>(0, 3) = cam_pos;
			                    // We will sample using nearest neighbour
			                    vuk::SamplerCreateInfo sci;
			                    sci.minFilter = sci.magFilter = vuk::Filter::eNearest;
			                    // Bind the previous attachments as sampled images
			                    command_buffer.bind_image(0, 0, "05_position+")
			                        .bind_sampler(0, 0, sci)
			                        .bind_image(0, 1, "05_normal+")
			                        .bind_sampler(0, 1, sci)
			                        .bind_image(0, 2, "05_color+")
			                        .bind_sampler(0, 2, sci)
			                        .draw(3, 1, 0, 0);
		                    } });

		      // The intermediate offscreen textures need to be bound
		      rg.attach_and_clear_image(
		          "05_position", { .format = vuk::Format::eR16G16B16A16Sfloat, .sample_count = vuk::Samples::e1 }, vuk::ClearColor{ 1.f, 0.f, 0.f, 0.f });
		      rg.attach_and_clear_image("05_normal", { .format = vuk::Format::eR16G16B16A16Sfloat }, vuk::ClearColor{ 0.f, 1.f, 0.f, 0.f });
		      rg.attach_and_clear_image("05_color", { .format = vuk::Format::eR8G8B8A8Srgb }, vuk::ClearColor{ 0.f, 0.f, 1.f, 0.f });
		      rg.attach_and_clear_image("05_depth", { .format = vuk::Format::eD32Sfloat }, vuk::ClearDepthStencil{ 1.0f, 0 });

		      // The framebuffer for the deferred rendering consists of "05_position", "05_normal", "05_color" and "05_depth" images
		      // Since these belong to the same framebuffer, vuk can infer the missing parameters that we don't explicitly provide
		      // For example all images in a framebuffer must be the same extent, hence it is enough to know the extent of one image
		      // In this case we have not specified any extent - and the second pass does not give any information
		      // So we provide an additional rule - the extent of "05_position" must match our target extent
		      // With this rule, all image parameters can be inferred
		      rg.inference_rule("05_position", vuk::same_extent_as("05_deferred"));

		      return vuk::Future{ std::make_unique<vuk::RenderGraph>(std::move(rg)), "05_deferred_final" };
		    },
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      verts.reset();
		      inds.reset();
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace