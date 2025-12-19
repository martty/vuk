#include "example_runner.hpp"

/* 13_tessellation
 * A simple demonstration of the tessellation pipeline in vuk.
 * This example shows how to use tessellation control and evaluation shaders to subdivide a basic triangle into smaller triangles.
 * The vertex shader generates same triangle used in 01_triangle example, which is then passed through the tessellation stages.
 *
 * These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
 * Furthermore it allows launching individual examples and all examples with the example same code.
 * Check out the framework (example_runner_*) files if interested!
 */

namespace {
	vuk::Example x{
		// The display name of this example
		.name = "13_tessellation",
		// Setup code, ran once in the beginning
		.setup =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& allocator, vuk::Runtime& runtime) {
		      vuk::PipelineBaseCreateInfo pci;
		      pci.add_glsl(util::read_entire_file((root / "examples/triangle.vert").generic_string()), (root / "examples/triangle.vert").generic_string());
		      pci.add_glsl(util::read_entire_file((root / "examples/triangle.frag").generic_string()), (root / "examples/triangle.frag").generic_string());
		      pci.add_glsl(util::read_entire_file((root / "examples/tess_tri.tesc").generic_string()), (root / "examples/tess_tri.tesc").generic_string());
		      pci.add_glsl(util::read_entire_file((root / "examples/tess_tri.tese").generic_string()), (root / "examples/tess_tri.tese").generic_string());
		      runtime.create_named_pipeline("tessellation", pci);
		    },
		// Code ran every frame
		.render =
		    [](vuk::ExampleRunner& runner, vuk::Allocator& frame_allocator, vuk::Value<vuk::ImageAttachment> target) {
		      auto pass = vuk::make_pass("13_tessellation", [](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eColorWrite) color_rt) {
			      command_buffer.set_viewport(0, vuk::Rect2D::framebuffer());
			      command_buffer.set_scissor(0, vuk::Rect2D::framebuffer());
			      command_buffer
			          .set_rasterization({ .polygonMode = vuk::PolygonMode::eLine }) //
			          .set_patch_control_points(3)
			          .set_primitive_topology(vuk::PrimitiveTopology::ePatchList)
			          .set_color_blend(color_rt, {})
			          .bind_graphics_pipeline("tessellation")
			          .draw(3, 1, 0, 0);
			      return color_rt;
		      });

		      auto drawn = pass(std::move(target));

		      return drawn;
		    }
	};

	REGISTER_EXAMPLE(x);
} // namespace
