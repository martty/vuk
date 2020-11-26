#include "example_runner.hpp"

/* 01_triangle
* In this example we will draw a bufferless triangle, the "Hello world" of graphics programming
* For this, we will need to define our pipeline, and then submit a draw.
*
* These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
* Furthermore it allows launching individual examples and all examples with the example same code.
* Check out the framework (example_runner_*) files if interested!
*/

namespace {
	vuk::Example x {
		// The display name of this example
		.name = "01_triangle",
		// Setup code, ran once in the beginning
		.setup = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			// Pipelines are created by filling out a vuk::PipelineCreateInfo
			// In this case, we only need the shaders, we don't care about the rest of the state
			vuk::PipelineBaseCreateInfo pci;
			pci.add_shader(util::read_entire_file("../../examples/triangle.vert"), "triangle.vert");
			pci.add_shader(util::read_entire_file("../../examples/triangle.frag"), "triangle.frag");
			// The pipeline is stored with a user give name for simplicity
			runner.context->create_named_pipeline("triangle", pci);
		},
		// Code ran every frame
		.render = [](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
			// We acquire a context specific to the thread we are on (PerThreadContext)
			auto ptc = ifc.begin();
			// We start building a rendergraph
			vuk::RenderGraph rg;
			// The rendergraph is composed of passes (vuk::Pass)
			// Each pass declares which resources are used
			// And it provides a callback which is executed when this pass is being ran
			rg.add_pass({
				// For this example, only a color image is needed to write to (our framebuffer)
				// The name is declared, and the way it will be used (color attachment - write)
				.resources = {"01_triangle_final"_image(vuk::eColorWrite)},
				.execute = [](vuk::CommandBuffer& command_buffer) {
					// Here commands are recorded into the command buffer for rendering
					// The commands frequently mimick the Vulkan counterpart, with additional sugar
					// The additional sugar is enabled by having a complete view of the rendering

					// Set the viewport to cover the entire framebuffer
					// By default the viewport is automatically flipped on Y to match OpenGL conventions
					command_buffer.set_viewport(0, vuk::Area::framebuffer());
					command_buffer
					  .set_scissor(0, vuk::Area::framebuffer()) // Set the scissor area to cover the entire framebuffer
					  .bind_graphics_pipeline("triangle") // Recall pipeline for "triangle" and bind
					  .draw(3, 1, 0, 0); // Draw 3 vertices
					}
				}
			);
			// The rendergraph is returned, where the example framework takes care of the busywork (submission, presenting)
			return rg;
		}
	};

	REGISTER_EXAMPLE(x);
}