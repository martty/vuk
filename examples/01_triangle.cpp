#include "example_runner.hpp"

vuk::Example x{
	.name = "01_triangle",
	.setup = [&](vuk::ExampleRunner& runner) {
		vuk::PipelineCreateInfo pci;
		pci.shaders.push_back("../../examples/triangle.vert");
		pci.shaders.push_back("../../examples/triangle.frag");
		pci.depth_stencil_state.depthCompareOp = vk::CompareOp::eAlways;
		runner.context->named_pipelines.emplace("triangle", pci);
	},
	.render = [&](vuk::ExampleRunner& runner, vuk::InflightContext& ifc) {
		auto ptc = ifc.begin();

		vuk::RenderGraph rg;
		rg.add_pass({
			.resources = {"01_triangle_final"_image(vuk::eColorWrite)},
			.execute = [&](vuk::CommandBuffer& command_buffer) {
				command_buffer
				  .set_viewport(0, vuk::Area::Framebuffer{})
				  .set_scissor(0, vuk::Area::Framebuffer{})
				  .bind_pipeline("triangle")
				  .draw(3, 1, 0, 0);
				}
			}
		);

		return rg;
	}
};

REGISTER_EXAMPLE(x);