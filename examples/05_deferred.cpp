#include "example_runner.hpp"
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

/* 05_deferred
* In this example we will take our cube to the next level by rendering it deferred.
* To achieve this, we will first render the cube to three offscreen textures -
* one containing the world position, the second the world normals and the third containing colour.
* We will also have depth buffering for this draw.
* After this, we will compute the shading by using a fullscreen pass, where we sample from these textures.
* To achieve this, we will need to let the rendergraph know of our image dependencies.
*
* These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
* Furthermore it allows launching individual examples and all examples with the example same code.
* Check out the framework (example_runner_*) files if interested!
*/

namespace {
	float angle = 0.f;
	auto box = util::generate_cube();

	vuk::Example x{
		.name = "05_deferred",
		.setup = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			{
			vuk::PipelineCreateInfo pci;
			pci.add_shader("../../examples/deferred.vert");
			pci.add_shader("../../examples/deferred.frag");
			runner.context->create_named_pipeline("cube_deferred", pci);
			}

			{
			vuk::PipelineCreateInfo pci;
			pci.add_shader("../../examples/fullscreen.vert");
			pci.add_shader("../../examples/deferred_resolve.frag");
			runner.context->create_named_pipeline("deferred_resolve", pci);
			}

		},
		.render = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			auto ptc = ifc.begin();

			// We set up the cube data, same as in example 02_cube

			auto [bverts, stub1] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eVertexBuffer, std::span(&box.first[0], box.first.size()));
			auto verts = std::move(bverts);
			auto [binds, stub2] = ptc.create_scratch_buffer(vuk::MemoryUsage::eGPUonly, vk::BufferUsageFlagBits::eIndexBuffer, std::span(&box.second[0], box.second.size()));
			auto inds = std::move(binds);
			struct VP {
				glm::mat4 view;
				glm::mat4 proj;
			} vp;
			auto cam_pos = glm::vec3(0, 1.5, 3.5);
			vp.view = glm::lookAt(cam_pos, glm::vec3(0), glm::vec3(0, 1, 0));
			vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);

			auto [buboVP, stub3] = ptc.create_scratch_buffer(vuk::MemoryUsage::eCPUtoGPU, vk::BufferUsageFlagBits::eUniformBuffer, std::span(&vp, 1));
			auto uboVP = buboVP;
			ptc.wait_all_transfers();

			vuk::RenderGraph rg;
			// Here we will render the cube into 3 offscreen textures
			rg.add_pass({
				// Passes can be optionally named, this useful for visualization and debugging
				.name = "05_deferred_MRT",
				// Declare our framebuffer
				.resources = {"05_position"_image(vuk::eColorWrite), "05_normal"_image(vuk::eColorWrite), "05_color"_image(vuk::eColorWrite), "05_depth"_image(vuk::eDepthStencilRW)},
				.execute = [verts, uboVP, inds](vuk::CommandBuffer& command_buffer) {
					// Rendering is the same as in the case for forward
					command_buffer
					  .set_viewport(0, vuk::Area::Framebuffer{})
					  .set_scissor(0, vuk::Area::Framebuffer{})
					  .bind_vertex_buffer(0, verts, 0, vuk::Packed{vk::Format::eR32G32B32Sfloat, vk::Format::eR32G32B32Sfloat, vuk::Ignore{offsetof(util::Vertex, uv_coordinates) - offsetof(util::Vertex, tangent)}, vk::Format::eR32G32Sfloat})
					  .bind_index_buffer(inds, vk::IndexType::eUint32)
					  .bind_pipeline("cube_deferred")
					  .bind_uniform_buffer(0, 0, uboVP);
					glm::mat4* model = command_buffer.map_scratch_uniform_binding<glm::mat4>(0, 1);
					*model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));
					command_buffer
					  .draw_indexed(box.second.size(), 1, 0, 0, 0);
					}
				}
			);

			angle += 360.f * ImGui::GetIO().DeltaTime;

			// The shading pass for the deferred rendering
			rg.add_pass({
				.name = "05_deferred_resolve",
				// Declare that we are going to render to the final color image
				// Declare that we are going to sample (in the fragment shader) from the previous attachments
				.resources = {"05_deferred_final"_image(vuk::eColorWrite), "05_position"_image(vuk::eFragmentSampled), "05_normal"_image(vuk::eFragmentSampled), "05_color"_image(vuk::eFragmentSampled)},
				.execute = [cam_pos](vuk::CommandBuffer& command_buffer) {
					command_buffer
					  .set_viewport(0, vuk::Area::Framebuffer{})
					  .set_scissor(0, vuk::Area::Framebuffer{})
					  .bind_pipeline("deferred_resolve");
					// Set camera position so we can do lighting
					*command_buffer.map_scratch_uniform_binding<glm::vec3>(0, 3) = cam_pos;
					// We will sample using nearest neighbour
					vk::SamplerCreateInfo sci;
					sci.minFilter = sci.magFilter = vk::Filter::eNearest;
					// Bind the previous attachments as sampled images
					command_buffer
					  .bind_sampled_image(0, 0, "05_position", sci)
					  .bind_sampled_image(0, 1, "05_normal", sci)
					  .bind_sampled_image(0, 2, "05_color", sci)
					  .draw(3, 1, 0, 0);
					}
				}
			);
			
			// The intermediate offscreen textures need to be bound
			// The "internal" rendering resolution is set here for one attachment, the rest infers from it
			rg.mark_attachment_internal("05_position", vk::Format::eR16G16B16A16Sfloat, vuk::Extent2D{300, 300}, vuk::Samples::e1, vuk::ClearColor{ 1.f,0.f,0.f,0.f });
			rg.mark_attachment_internal("05_normal", vk::Format::eR16G16B16A16Sfloat, vuk::Extent2D::Framebuffer{}, vuk::Samples::e1, vuk::ClearColor{ 0.f, 1.f, 0.f, 0.f });
			rg.mark_attachment_internal("05_color", vk::Format::eR8G8B8A8Unorm, vuk::Extent2D::Framebuffer{}, vuk::Samples::e1, vuk::ClearColor{ 0.f, 0.f, 1.f, 0.f });
			rg.mark_attachment_internal("05_depth", vk::Format::eD32Sfloat, vuk::Extent2D::Framebuffer{}, vuk::Samples::Framebuffer{}, vuk::ClearDepthStencil{ 1.0f, 0 });

			return rg;
		}
	};

	REGISTER_EXAMPLE(x);
}