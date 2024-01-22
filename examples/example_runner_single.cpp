#include "example_runner.hpp"

void vuk::ExampleRunner::render() {
	Compiler compiler;
	// the examples can all enqueue upload tasks via enqueue_setup. for simplicity, we submit and wait for all the upload tasks before moving on to the render
	// loop in a real application, one would have something more complex to handle uploading data it is also possible to wait for the uploads on the GPU by using
	// these uploading futures as input
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
		// all of the objects allocated from this allocator last for this frame, and get recycled automatically, so for this specific allocator, deallocation is
		// optional
		Allocator frame_allocator(frame_resource);
		// create a rendergraph we will use to prepare a swapchain image for the example to render into
		auto imported_swapchain = declare_swapchain(*swapchain);
		// acquire an image on the swapchain
		auto swapchain_image = acquire_next_image("swp_img", std::move(imported_swapchain));

		// clear the swapchain image
		Value<ImageAttachment> cleared_image_to_render_into = clear_image(std::move(swapchain_image), vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		// invoke the render method of the example with the cleared image
		Value<ImageAttachment> example_result = examples[0]->render(*this, frame_allocator, std::move(cleared_image_to_render_into));

		// set up some profiling callbacks for our example Tracy integration
		vuk::ProfilingCallbacks cbs;
		cbs.user_data = &get_runner();
		cbs.on_begin_command_buffer = [](void* user_data, VkCommandBuffer cbuf) {
			ExampleRunner& runner = *reinterpret_cast<vuk::ExampleRunner*>(user_data);
			TracyVkCollect(runner.tracy_graphics_ctx, cbuf);
			TracyVkCollect(runner.tracy_transfer_ctx, cbuf);
			return (void*)nullptr;
		};
		// runs whenever entering a new vuk::Pass
		// we start a GPU zone and then keep it open
		cbs.on_begin_pass = [](void* user_data, Name pass_name, VkCommandBuffer cmdbuf, DomainFlagBits domain) {
			ExampleRunner& runner = *reinterpret_cast<vuk::ExampleRunner*>(user_data);
			void* pass_data = new char[sizeof(tracy::VkCtxScope)];
			if (domain & vuk::DomainFlagBits::eGraphicsQueue) {
#if defined TRACY_ENABLE
				new (pass_data) TracyVkZoneTransient(runner.tracy_graphics_ctx, , cmdbuf, pass_name.c_str(), true);
#endif
			} else if (domain & vuk::DomainFlagBits::eTransferQueue) {
#if defined TRACY_ENABLE
				new (pass_data) TracyVkZoneTransient(runner.tracy_transfer_ctx, , cmdbuf, pass_name.c_str(), true);
#endif
			}

			return pass_data;
		};
		// runs whenever a pass has ended, we end the GPU zone we started
		cbs.on_end_pass = [](void* user_data, void* pass_data) {
			auto tracy_scope = reinterpret_cast<tracy::VkCtxScope*>(pass_data);
#if defined TRACY_ENABLE
			tracy_scope->~VkCtxScope();
#endif
			delete reinterpret_cast<char*>(pass_data);
		};

		// compile the RG that contains all the rendering of the example
		// submit and present the results to the swapchain we imported previously
		auto entire_thing = enqueue_presentation(std::move(example_result));

		entire_thing.wait(frame_allocator, compiler, { .callbacks = cbs });

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
	auto path_to_root = std::filesystem::relative(VUK_EX_PATH_ROOT, VUK_EX_PATH_TGT);
	root = std::filesystem::canonical(std::filesystem::path(argv[0]).parent_path() / path_to_root);
	// very simple error handling in the example framework: we don't check for errors and just let them be converted into exceptions that are caught at top level
	try {
		vuk::ExampleRunner::get_runner().setup();
		vuk::ExampleRunner::get_runner().render();
		vuk::ExampleRunner::get_runner().cleanup();
	} catch (vuk::Exception& e) {
		fprintf(stderr, "%s", e.what());
	}
}
