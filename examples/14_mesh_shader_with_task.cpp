#include "example_runner.hpp"

/* 14_mesh_shader_with_task
 * This example demonstrates the full mesh shader pipeline including task shaders.
 * Task shaders can be used to cull or amplify mesh shader workgroups before they execute.
 * This example uses a task shader to dispatch a mesh shader workgroup that draws a triangle.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	vuk::Example x{
		// The display name of this example
		.name = "14_mesh_shader_with_task",
		// Setup code, ran once in the beginning
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator, vuk::Runtime& runtime) {
		      // Create a pipeline with task, mesh, and fragment shaders
		      vuk::PipelineBaseCreateInfo pci;
		      pci.add_glsl(util::read_entire_file((root / "examples/triangle.task").generic_string()), (root / "examples/triangle.task").generic_string());
		      pci.add_glsl(util::read_entire_file((root / "examples/triangle.mesh").generic_string()), (root / "examples/triangle.mesh").generic_string());
		      pci.add_glsl(util::read_entire_file((root / "examples/triangle.frag").generic_string()), (root / "examples/triangle.frag").generic_string());
		      runtime.create_named_pipeline("mesh_shader_with_task", pci);
		    },
		// Code ran every frame
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Value<vuk::ImageAttachment> target) {
		      auto pass = vuk::make_pass("14_mesh_shader_with_task", [](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eColorWrite) color_rt) {
			      command_buffer.set_viewport(0, vuk::Rect2D::framebuffer());
			      command_buffer.set_scissor(0, vuk::Rect2D::framebuffer());
			      command_buffer
			          .set_rasterization({})
			          .set_color_blend(color_rt, {})
			          .bind_graphics_pipeline("mesh_shader_with_task")
			          .draw_mesh_tasks(1, 1, 1); // Draw 1 task shader workgroup
			      return color_rt;
		      });

		      auto drawn = pass(std::move(target));

		      return drawn;
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace
