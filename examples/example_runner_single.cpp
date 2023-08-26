#include "example_runner.hpp"

vuk::SingleSwapchainRenderBundle bundle;

void vuk::ExampleRunner::render() {
	Compiler compiler;
	// the examples can all enqueue upload tasks via enqueue_setup. for simplicity, we submit and wait for all the upload tasks before moving on to the render loop
	// in a real application, one would have something more complex to handle uploading data
	// it is also possible to wait for the uploads on the GPU by using these uploading futures as input
	vuk::wait_for_futures_explicit(*superframe_allocator, compiler, futures);
	futures.clear();

	// our main loop
	while (!glfwWindowShouldClose(window)) {
		// pump the message loop
		glfwPollEvents();
		while (suspend) {
			glfwWaitEvents();
		}
		// advance the frame for the allocators and caches used by vuk
		auto& frame_resource = superframe_resource->get_next_frame();
		context->next_frame();
		// create a frame allocator - we can allocate objects for the duration of the frame from this allocator
		// all of the objects allocated from this allocator last for this frame, and get recycled automatically, so for this specific allocator, deallocation is optional
		Allocator frame_allocator(frame_resource);
		// acquire an image on the swapchain
		bundle = *acquire_one(*context, swapchain, (*present_ready)[context->get_frame_count() % 3], (*render_complete)[context->get_frame_count() % 3]);
		// create a rendergraph we will use to prepare a swapchain image for the example to render into
		std::shared_ptr<RenderGraph> rg(std::make_shared<RenderGraph>("runner"));
		// we bind the swapchain to name "_swp"
		rg->attach_swapchain("_swp", swapchain);
		// clear the "_swp" image and call the cleared image "example_target_image"
		rg->clear_image("_swp", "example_target_image", vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		// bind "example_target_image" as the output of this rendergraph
		Future cleared_image_to_render_into{ std::move(rg), "example_target_image" };
		// invoke the render method of the example with the cleared image
		Future example_result = examples[0]->render(*this, frame_allocator, std::move(cleared_image_to_render_into));
		// make a new RG that will take care of putting the swapchain image into present and releasing it from the rg
		std::shared_ptr<RenderGraph> rg_p(std::make_shared<RenderGraph>("presenter"));
		rg_p->attach_in("_src", std::move(example_result));
		// we tell the rendergraph that _src will be used for presenting after the rendergraph
		rg_p->release_for_present("_src");
		// compile the RG that contains all the rendering of the example
		auto erg = *compiler.link(std::span{ &rg_p, 1 }, {});
		// submit the compiled commands
		auto result = *execute_submit(frame_allocator, std::move(erg), std::move(bundle));
		// present the results
		present_to_one(*context, std::move(result));
		// update window title with FPS
		if (++num_frames == 16) {
			auto new_time = get_time();
			auto delta = new_time - old_time;
			auto per_frame_time = delta / 16 * 1000;
			old_time = new_time;
			num_frames = 0;
			set_window_title(std::string("Vuk example [") + std::to_string(per_frame_time) + " ms / " + std::to_string(1000 / per_frame_time) + " FPS]");
		}
	}
}

int main(int argc, char** argv) {
	root = std::filesystem::canonical(std::filesystem::path(argv[0]).parent_path() / VUK_EX_PATH_TO_ROOT);
	// very simple error handling in the example framework: we don't check for errors and just let them be converted into exceptions that are caught at top level
	try {
		vuk::ExampleRunner::get_runner().setup();
		vuk::ExampleRunner::get_runner().render();
		vuk::ExampleRunner::get_runner().cleanup();
	} catch (vuk::Exception& e) {
		fprintf(stderr, "%s", e.what());
	}
}
