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
	vuk::Example x{
		// The display name of this example
		.name = "01_triangle_vcc",
		// Setup code, ran once in the beginning
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator) {
		      // Pipelines are created by filling out a vuk::PipelineCreateInfo
		      // In this case, we only need the shaders, we don't care about the rest of the state
		      vuk::PipelineBaseCreateInfo pci;
		      pci.add_glsl(util::read_entire_file((root / "examples/triangle.vert").generic_string()), (root / "examples/triangle.vert").generic_string());
		      pci.add_c(util::read_entire_file((root / "examples/triangle.frag.c").generic_string()), (root / "examples/triangle.frag.c").generic_string());
		      // The pipeline is stored with a user give name for simplicity
		      runner.context->create_named_pipeline("triangle", pci);
		    },
		// Code ran every frame
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Future target) {
		      // We start building a rendergraph
		      vuk::RenderGraph rg("01");
		      // The framework provides us with an image to render to in "target"
		      // We attach this to the rendergraph named as "01_triangle"
		      rg.attach_in("01_triangle", std::move(target));
		      // The rendergraph is composed of passes (vuk::Pass)
		      // Each pass declares which resources are used
		      // And it provides a callback which is executed when this pass is being ran
		      rg.add_pass({ // For this example, only a color image is needed to write to (our framebuffer)
		                    // The name is declared, and the way it will be used in the pass (color attachment - write)
		                    .resources = { "01_triangle"_image >> vuk::eColorWrite >> "01_triangle_final" },
		                    .execute = [](vuk::CommandBuffer& command_buffer) {
			                    // Here commands are recorded into the command buffer for rendering
			                    // The commands frequently mimick the Vulkan counterpart, with additional sugar
			                    // The additional sugar is enabled by having a complete view of the rendering

			                    // Set the viewport to cover the entire framebuffer
			                    command_buffer.set_viewport(0, vuk::Rect2D::framebuffer());
			                    // Set the scissor area to cover the entire framebuffer
			                    command_buffer.set_scissor(0, vuk::Rect2D::framebuffer());
			                    command_buffer
			                        .set_rasterization({})              // Set the default rasterization state
			                        .set_color_blend("01_triangle", {}) // Set the default color blend state
			                        .bind_graphics_pipeline("triangle") // Recall pipeline for "triangle" and bind
			                        .draw(3, 1, 0, 0);                  // Draw 3 vertices
		                    } });

		      // The rendergraph is given to a Future, which takes ownership and binds to the result ("01_triangle_final")
		      // The example framework takes care of the busywork (submission, presenting)
		      return vuk::Future{ std::make_unique<vuk::RenderGraph>(std::move(rg)), "01_triangle_final" };
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace