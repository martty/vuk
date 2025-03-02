#include "example_runner.hpp"

void vuk::ExampleRunner::render() {
	Compiler compiler;
	// the examples can all enqueue upload tasks via enqueue_setup. for simplicity, we submit and wait for all the upload tasks before moving on to the render
	// loop in a real application, one would have something more complex to handle uploading data it is also possible to wait for the uploads on the GPU by using
	// these uploading futures as input
	vuk::wait_for_values_explicit(*app->superframe_allocator, compiler, futures);
	futures.clear();

	// our main loop
	while (!glfwWindowShouldClose(window)) {
		// pump the message loop
		glfwPollEvents();
		while (suspend) {
			glfwWaitEvents();
		}
		// advance the frame for the allocators and caches used by vuk
		auto& frame_resource = app->superframe_resource->get_next_frame();
		app->next_frame();
		// create a frame allocator - we can allocate objects for the duration of the frame from this allocator
		// all of the objects allocated from this allocator last for this frame, and get recycled automatically, so for this specific allocator, deallocation is
		// optional
		Allocator frame_allocator(frame_resource);
		// create a rendergraph we will use to prepare a swapchain image for the example to render into
		auto imported_swapchain = acquire_swapchain(*app->swapchain);
		// acquire an image on the swapchain
		auto swapchain_image = acquire_next_image("swp_img", std::move(imported_swapchain));

		// clear the swapchain image
		Value<ImageAttachment> cleared_image_to_render_into = clear_image(std::move(swapchain_image), vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		// invoke the render method of the example with the cleared image
		Value<ImageAttachment> example_result = examples[0]->render(*this, frame_allocator, std::move(cleared_image_to_render_into));

		vuk::ProfilingCallbacks cbs;
#ifdef TRACY_ENABLE
		// set up some profiling callbacks for our Tracy integration
		cbs = make_Tracy_callbacks(*tracy_context);
#endif

		// compile the IRModule that contains all the rendering of the example
		// submit and present the results to the swapchain we imported previously
		auto entire_thing = enqueue_presentation(std::move(example_result));

		entire_thing.submit(frame_allocator, compiler, { .callbacks = cbs });

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