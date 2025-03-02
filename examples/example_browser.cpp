#include "example_runner.hpp"

bool render_all = true;

void vuk::ExampleRunner::render() {
	Compiler compiler;

	vuk::wait_for_values_explicit(*app->superframe_allocator, compiler, futures);
	futures.clear();

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		while (suspend) {
			glfwWaitEvents();
		}
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 352.f, 2));
		ImGui::SetNextWindowSize(ImVec2(350, 0));
		ImGui::Begin("Example selector", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);
		ImGui::Checkbox("All", &render_all);
		ImGui::SameLine();

		static vuk::Example* item_current = examples[7]; // Here our selection is a single pointer stored outside the object.
		if (!render_all) {
			if (ImGui::BeginCombo("Examples", item_current->name.data(), ImGuiComboFlags_None)) {
				for (int n = 0; n < examples.size(); n++) {
					bool is_selected = (item_current == examples[n]);
					if (ImGui::Selectable(examples[n]->name.data(), is_selected))
						item_current = examples[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus(); // Set the initial focus when opening the combo (scrolling + for keyboard navigation support in the upcoming
						                              // navigation branch)
				}
				ImGui::EndCombo();
			}
		}
		ImGui::End();

		auto& frame_resource = app->superframe_resource->get_next_frame();
		app->next_frame();

		Allocator frame_allocator(frame_resource);

		auto imported_swapchain = acquire_swapchain(*app->swapchain);
		// acquire an image on the swapchain
		auto swapchain_image = acquire_next_image("swp_img", std::move(imported_swapchain));
		// clear the swapchain image
		Value<ImageAttachment> cleared_image_to_render_into = clear_image(std::move(swapchain_image), vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });

		Value<ImageAttachment> imgui;
		if (!render_all) { // render a single full window example
			Value<ImageAttachment> example_result = item_current->render(*this, frame_allocator, std::move(cleared_image_to_render_into));

			ImGui::Render();

			imgui = extra::ImGui_ImplVuk_Render(frame_allocator, std::move(example_result), imgui_data);
		} else { // render all examples as imgui windows
			for (size_t i = 0; i < examples.size(); i++) {
				auto& ex = examples[i];
				ImGui::SetNextWindowSize(ImVec2(250, 250), ImGuiCond_FirstUseEver);
				ImGui::SetNextWindowPos(ImVec2((float)(i % 4) * 250, ((float)i / 4) * 250), ImGuiCond_FirstUseEver);
				ImGui::Begin(ex->name.data());
				auto size = ImGui::GetContentRegionAvail();
				size.x = size.x <= 0 ? 1 : size.x;
				size.y = size.y <= 0 ? 1 : size.y;
				auto small_target = vuk::clear_image(vuk::declare_ia("_img",
				                                                     { .extent = { (uint32_t)size.x, (uint32_t)size.y, 1 },
				                                                       .format = app->swapchain->images[0].format,
				                                                       .sample_count = vuk::Samples::e1,
				                                                       .level_count = 1,
				                                                       .layer_count = 1 }),
				                                     vuk::ClearColor(0.1f, 0.2f, 0.3f, 1.f));
				auto rendered_image = ex->render(*this, frame_allocator, std::move(small_target));

				ImGui::Image(imgui_data.add_image(std::move(rendered_image)), ImGui::GetContentRegionAvail());
				ImGui::End();
			}

			ImGui::Render();

			imgui = extra::ImGui_ImplVuk_Render(frame_allocator, std::move(cleared_image_to_render_into), imgui_data);
		}

		// compile the IRModule that contains all the rendering of the example
		// submit and present the results to the swapchain we imported previously
		auto entire_thing = enqueue_presentation(std::move(imgui));

		vuk::ProfilingCallbacks cbs;
#ifdef TRACY_ENABLE
		// set up some profiling callbacks for our Tracy integration
		cbs = make_Tracy_callbacks(*tracy_context);
#endif

		entire_thing.submit(frame_allocator, compiler, {.callbacks = cbs});

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