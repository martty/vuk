#include "example_runner.hpp"

vuk::SingleSwapchainRenderBundle bundle;

void vuk::ExampleRunner::render() {
	Compiler compiler;
	vuk::wait_for_futures_explicit(*global, compiler, futures);
	futures.clear();

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		while (suspend) {
			glfwWaitEvents();
		}
		auto& xdev_frame_resource = xdev_rf_alloc->get_next_frame();
		context->next_frame();
		Allocator frame_allocator(xdev_frame_resource);
		bundle = *acquire_one(*context, swapchain, (*present_ready)[context->get_frame_count() % 3], (*render_complete)[context->get_frame_count() % 3]);
		RenderGraph rg("runner");
		auto attachment_name = vuk::Name(examples[0]->name);
		rg.attach_swapchain("_swp", swapchain);
		rg.clear_image("_swp", attachment_name, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		auto fut = examples[0]->render(*this, frame_allocator, Future{ std::make_shared<RenderGraph>(std::move(rg)), attachment_name });
		auto ptr = fut.get_render_graph();
		auto erg = *compiler.link(std::span{ &ptr, 1 }, {});
		auto result = *execute_submit(frame_allocator, std::move(erg), std::move(bundle));
		present_to_one(*context, std::move(result));
		if (++num_frames == 16) {
			auto new_time = get_time();
			auto delta = new_time - old_time;
			auto per_frame_time = delta / 16 * 1000;
			old_time = new_time;
			num_frames = 0;
			set_window_title(std::string("Vuk example browser [") + std::to_string(per_frame_time) + " ms / " + std::to_string(1000 / per_frame_time) + " FPS]");
		}
	}
}

int main() {
	vuk::ExampleRunner::get_runner().setup();
	vuk::ExampleRunner::get_runner().render();
	vuk::ExampleRunner::get_runner().cleanup();
}
