#include "example_runner.hpp"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/mat4x4.hpp>

/* 02_cube
 * In this example we will draw a cube, still with a single attachment, but using vertex, index and uniform buffers.
 * The cube will spin around its Y axis, which we will achieve by changing the model matrix each frame.
 * This examples showcases using scratch allocations, which only live for one frame.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	// The Y rotation angle of our cube
	float angle = 0.f;
	// Generate vertices and indices for the cube
	auto box = util::generate_cube();
	vuk::Unique<vuk::Buffer> verts, inds;

	vuk::Example x{
		.name = "02_cube",
		// Same setup as previously
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      vuk::PipelineBaseCreateInfo pci;
		      pci.add_glsl(util::read_entire_file((root / "examples/ubo_test.vert").generic_string()), (root / "examples/ubo_test.vert").generic_string());
		      pci.add_glsl(util::read_entire_file((root / "examples/triangle_depthshaded.frag").generic_string()),
		                   (root / "examples/triangle_depthshaded.frag").generic_string());
		      pci.define("SCALE", "0.75");
		      allocator.get_context().create_named_pipeline("cube", pci);

		      // Request a GPU-only buffer allocation with specific data
		      // The buffer is allocated in device-local, non-host visible memory
		      // And enqueues a transfer operation on the graphics queue, which will copy the given data
		      // Finally it returns a vuk::Buffer, which holds the info for the allocation and a Future that represents the upload being completed
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
		      // This struct will represent the view-projection transform used for the cube
		      struct VP {
			      glm::mat4 view;
			      glm::mat4 proj;
		      } vp;
		      // Fill the view matrix, looking a bit from top to the center
		      vp.view = glm::lookAt(glm::vec3(0, 1.5, 3.5), glm::vec3(0), glm::vec3(0, 1, 0));
		      // Fill the projection matrix, standard perspective matrix
		      vp.proj = glm::perspective(glm::degrees(70.f), 1.f, 1.f, 10.f);
		      vp.proj[1][1] *= -1;
		      // Allocate and transfer view-projection transform
		      auto [buboVP, uboVP_fut] = create_buffer(frame_allocator, vuk::MemoryUsage::eCPUtoGPU, vuk::DomainFlagBits::eTransferOnGraphics, std::span(&vp, 1));
		      // since this memory is CPU visible (MemoryUsage::eCPUtoGPU), we don't need to wait for the future to complete
		      auto uboVP = *buboVP;

		      vuk::RenderGraph rg("02");
		      rg.attach_in("02_cube", std::move(target));
		      rg.add_pass(
		          { // For this example, only a color image is needed to write to (our framebuffer)
		            // The name is declared, and the way it will be used in the pass (color attachment - write)
		            .resources = { "02_cube"_image >> vuk::eColorWrite >> "02_cube_final" },
		            .execute = [uboVP](vuk::CommandBuffer& command_buffer) {
			            command_buffer
			                // In vuk, all pipeline state (with the exception of the shaders) come from the command buffer
			                // Such state can be requested to be dynamic - dynamic state does not form part of the pipeline key, and hence cheap to change
			                // On desktop, dynamic scissor and viewport is of no extra cost, and worthwhile to set always
			                .set_dynamic_state(vuk::DynamicStateFlagBits::eScissor | vuk::DynamicStateFlagBits::eViewport)
			                .set_viewport(0, vuk::Rect2D::framebuffer()) // Set the viewport to cover the entire framebuffer
			                .set_scissor(0, vuk::Rect2D::framebuffer())  // Set the scissor area to cover the entire framebuffer
			                .set_rasterization({})                       // Set the default rasterization state
			                .broadcast_color_blend({})                   // Set the default color blend state
			                // The vertex format and the buffer used are bound together for this call
			                // The format is specified here as vuk::Packed{}, meaning we are going to make a consecutive binding
			                // For each element in the list, a vuk::Format signifies a binding
			                // And a vuk::Ignore signifies a number of bytes to be skipped
			                // In this case, we will bind vuk::Format::eR32G32B32Sfloat to the first location (0)
			                // And use the remaining vuk::Ignore-d bytes to establish the stride of the buffer
			                .bind_vertex_buffer(
			                    0, *verts, 0, vuk::Packed{ vuk::Format::eR32G32B32Sfloat, vuk::Ignore{ sizeof(util::Vertex) - sizeof(util::Vertex::position) } })
			                // Bind the index buffer
			                .bind_index_buffer(*inds, vuk::IndexType::eUint32)
			                .bind_graphics_pipeline("cube")
			                // Bind the uniform buffer we allocated to (set = 0, binding = 0)
			                .bind_buffer(0, 0, uboVP)
			                .bind_buffer(0, 1, uboVP); // It is allowed to bind to slots that are not consumed by the current pipeline
			            // For the model matrix, we will take a shorter route
			            // Frequently updated uniform buffers should be in CPUtoGPU type memory, which is mapped
			            // So we create a typed mapping directly and write the model matrix
			            glm::mat4* model = command_buffer.map_scratch_buffer<glm::mat4>(0, 1);
			            *model = static_cast<glm::mat4>(glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)));

			            // We can also customize pipelines by using specialization constants
			            // Here we will apply a tint based on the current frame
			            auto current_frame = command_buffer.get_context().get_frame_count();
			            auto mod_frame = current_frame % 1000;
			            glm::vec3 tint{ 1.f, 1.f, 1.f };
			            if (mod_frame <= 500 && mod_frame > 250) {
				            tint = { 1.f, 0.5f, 0.5f };
			            } else if (mod_frame <= 750 && mod_frame > 500) {
				            tint = { 0.5f, 1.0f, 0.5f };
			            } else if (mod_frame > 750) {
				            tint = { 0.5f, 0.5f, 1.0f };
			            }
			            // Specialization constants can only be scalars, use three to make a vec3
			            command_buffer.specialize_constants(0, tint.x).specialize_constants(1, tint.y).specialize_constants(2, tint.z);
			            // The cube is drawn via indexed drawing
			            command_buffer.draw_indexed(box.second.size(), 1, 0, 0, 0);
		            } });
		      // The angle is update to rotate the cube
		      angle += 20.f * ImGui::GetIO().DeltaTime;

		      return vuk::Future{ std::make_unique<vuk::RenderGraph>(std::move(rg)), "02_cube_final" };
		    },
		.cleanup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator) {
		      verts.reset();
		      inds.reset();
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace