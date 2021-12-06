#include "bench_runner.hpp"

/* 01_triangle
* In this example we will draw a bufferless triangle, the "Hello world" of graphics programming
* For this, we will need to define our pipeline, and then submit a draw.
*
* These examples are powered by the example framework, which hides some of the code required, as that would be repeated for each example.
* Furthermore it allows launching individual examples and all examples with the example same code.
* Check out the framework (example_runner_*) files if interested!
*/

namespace {
	struct V1 {
		std::string_view description = "1 iter";
		static constexpr unsigned n_iters = 1;
	};

	struct V2 {
		std::string_view description = "100 iters";
		static constexpr unsigned n_iters = 100;
	};

	vuk::Bench<V1, V2> x{
		// The display name of this example
		.base = {
			.name = "Dependent vs. non-dependent texture fetch",
			// Setup code, ran once in the beginning
			.setup = [](vuk::BenchRunner& runner, vuk::Allocator& frame_allocator) {
			// Pipelines are created by filling out a vuk::PipelineCreateInfo
			// In this case, we only need the shaders, we don't care about the rest of the state
			vuk::PipelineBaseCreateInfo pci;
			pci.add_glsl(util::read_entire_file("../../examples/triangle.vert"), "triangle.vert");
			pci.add_glsl(util::read_entire_file("../../examples/triangle.frag"), "triangle.frag");
			// The pipeline is stored with a user give name for simplicity
			runner.context->create_named_pipeline("triangle", pci);
			},
			.gui = [](vuk::BenchRunner& runner, vuk::Allocator& frame_allocator) {
			}
		},
		.cases = {
			{"Dependent, small image", [](vuk::BenchRunner& runner, vuk::Allocator& frame_allocator, vuk::Query start, vuk::Query end, auto&& parameters) {
			auto ptc = ifc.begin();
			vuk::RenderGraph rg;
			rg.add_pass({
				.resources = {"_final"_image(vuk::eColorWrite)},
				.execute = [start, end, parameters](vuk::CommandBuffer& command_buffer) {
					vuk::TimedScope _{command_buffer, start, end};
					command_buffer.set_viewport(0, vuk::Rect2D::framebuffer());
					command_buffer
					  .set_scissor(0, vuk::Rect2D::framebuffer()) // Set the scissor area to cover the entire framebuffer
					  .bind_graphics_pipeline("triangle") // Recall pipeline for "triangle" and bind
					  .draw(3 * parameters.n_iters, 1, 0, 0); // Draw 3 vertices
					}
				}
			);
			return rg;
		}},
				{"Non-dependent, small image", [](vuk::BenchRunner& runner, vuk::Allocator& frame_allocator, vuk::Query start, vuk::Query end, auto&& parameters) {
			auto ptc = ifc.begin();
			vuk::RenderGraph rg;
			rg.add_pass({
				.resources = {"_final"_image(vuk::eColorWrite)},
				.execute = [start, end, parameters](vuk::CommandBuffer& command_buffer) {
					vuk::TimedScope _{command_buffer, start, end};
					command_buffer.set_viewport(0, vuk::Rect2D::framebuffer());
					command_buffer.set_scissor(0, vuk::Rect2D::framebuffer())
						.bind_graphics_pipeline("triangle");
					for (auto i = 0; i < parameters.n_iters; i++) {
						command_buffer.draw(3, 1, 0, 0);
					}
					}
				}
			);
			return rg;
		}}
		}
	};

	REGISTER_BENCH(x);
}