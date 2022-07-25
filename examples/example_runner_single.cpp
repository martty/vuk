#include "example_runner.hpp"

void vuk::ExampleRunner::render() {
	Compiler compiler;
	vuk::wait_for_futures_explicit(*global, compiler, futures);
	futures.clear();

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		auto& xdev_frame_resource = xdev_rf_alloc->get_next_frame();
		context->next_frame();
		Allocator frame_allocator(xdev_frame_resource);
		RenderGraph rg("runner");
		auto attachment_name = vuk::Name(examples[0]->name);
		rg.attach_swapchain("_swp", swapchain);
		rg.clear_image("_swp", attachment_name, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		auto fut = examples[0]->render(*this, frame_allocator, Future{ std::make_shared<RenderGraph>(std::move(rg)), attachment_name });
		present(frame_allocator, compiler, swapchain, std::move(fut));
	}
}

int main() {
	vuk::ExampleRunner::get_runner().setup();
	vuk::ExampleRunner::get_runner().render();
	vuk::ExampleRunner::get_runner().cleanup();
}
