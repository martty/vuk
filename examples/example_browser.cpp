#include "../src/RenderGraphUtil.hpp"
#include "example_runner.hpp"

std::vector<vuk::Name> chosen_resource;

bool render_all = true;
vuk::SingleSwapchainRenderBundle bundle;

void vuk::ExampleRunner::render() {
	Compiler compiler;
	chosen_resource.resize(examples.size());

	vuk::wait_for_futures_explicit(*global, compiler, futures);
	futures.clear();

	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
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

		auto& xdev_frame_resource = xdev_rf_alloc->get_next_frame();
		context->next_frame();

		Allocator frame_allocator(xdev_frame_resource);
		if (!render_all) { // render a single full window example
			RenderGraph rg("runner");
			vuk::Name attachment_name = item_current->name;
			rg.attach_swapchain("_swp", swapchain);
			rg.clear_image("_swp", attachment_name, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
			auto fut = item_current->render(*this, frame_allocator, Future{ std::make_shared<RenderGraph>(std::move(rg)), attachment_name });
			ImGui::Render();

			fut = util::ImGui_ImplVuk_Render(frame_allocator, std::move(fut), imgui_data, ImGui::GetDrawData(), sampled_images);
			auto ptr = fut.get_render_graph();
			auto erg = *compiler.link(std::span{ &ptr, 1 }, {});
			bundle = *acquire_one(*context, swapchain, (*present_ready)[context->get_frame_count() % 3], (*render_complete)[context->get_frame_count() % 3]);
			auto result = *execute_submit(frame_allocator, std::move(erg), std::move(bundle));
			present_to_one(*context, std::move(result));
			sampled_images.clear();
		} else { // render all examples as imgui windows
			std::shared_ptr<RenderGraph> rg = std::make_shared<RenderGraph>("runner");
			plf::colony<vuk::Name> attachment_names;

			size_t i = 0;
			for (auto& ex : examples) {
				std::shared_ptr<RenderGraph> rgx = std::make_shared<RenderGraph>(ex->name);
				ImGui::SetNextWindowSize(ImVec2(250, 250), ImGuiCond_FirstUseEver);
				ImGui::SetNextWindowPos(ImVec2((float)(i % 4) * 250, ((float)i / 4) * 250), ImGuiCond_FirstUseEver);
				ImGui::Begin(ex->name.data());
				auto size = ImGui::GetContentRegionAvail();
				size.x = size.x <= 0 ? 1 : size.x;
				size.y = size.y <= 0 ? 1 : size.y;
				rgx->attach_and_clear_image("_img",
				                            { .extent = vuk::Dimension3D::absolute((uint32_t)size.x, (uint32_t)size.y),
				                              .format = swapchain->format,
				                              .sample_count = vuk::Samples::e1,
				                              .level_count = 1,
				                              .layer_count = 1 },
				                            vuk::ClearColor(0.1f, 0.2f, 0.3f, 1.f));
				auto rg_frag_fut = ex->render(*this, frame_allocator, Future{ rgx, "_img" });
				Name& attachment_name_out = *attachment_names.emplace(std::string(ex->name) + "_final");
				auto rg_frag = rg_frag_fut.get_render_graph();
				compiler.compile({ &rg_frag, 1 }, {});
				if (auto use_chains = compiler.get_use_chains(); use_chains.size() > 1) {
					const auto& bound_attachments = compiler.get_bound_attachments();
					bool disable = false;
					for (const auto [key, use_refs] : use_chains) {
						auto bound_it = bound_attachments.find(key);
						if (bound_it == bound_attachments.end())
							continue;
						auto samples = vuk::SampleCountFlagBits::e1;
						auto& att_info = (*bound_it).second;
						if (att_info.attachment.sample_count != vuk::Samples::eInfer)
							samples = att_info.attachment.sample_count.count;
						disable = disable || (samples != vuk::SampleCountFlagBits::e1);
					}

					for (const auto [key, use_refs] : use_chains) {
						auto bound_it = bound_attachments.find(key);
						if (bound_it == bound_attachments.end())
							continue;
						std::string btn_id = "";
						bool prevent_disable = false;
						if (key.to_sv() == attachment_name_out) {
							prevent_disable = true;
							btn_id = "F";
						} else {
							auto usage = compiler.compute_usage(use_refs);
							if (usage & vuk::ImageUsageFlagBits::eColorAttachment) {
								btn_id += "C";
							} else if (usage & vuk::ImageUsageFlagBits::eDepthStencilAttachment) {
								btn_id += "D";
							} else if (usage & (vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst)) {
								btn_id += "X";
							}
						}
						if (disable && !prevent_disable) {
							btn_id += " (MS)";
						} else {
							btn_id += "##" + std::string(key.to_sv());
						}
						if (disable && !prevent_disable) {
							ImGui::TextDisabled("%s", btn_id.c_str());
						} else {
							if (ImGui::Button(btn_id.c_str())) {
								if (key.to_sv() == ex->name) {
									chosen_resource[i] = attachment_name_out;
								} else {
									Name last_use = use_refs.back().out_name.is_invalid() ? use_refs.back().name : use_refs.back().out_name;
									auto sv = last_use.to_sv();
									sv.remove_prefix(rg_frag->name.to_sv().size() + 2);
									chosen_resource[i] = sv;
								}
							}
						}
						if (ImGui::IsItemHovered())
							ImGui::SetTooltip("%s", key.c_str());
						ImGui::SameLine();
					}
					ImGui::NewLine();
				}
				if (chosen_resource[i].is_invalid())
					chosen_resource[i] = attachment_name_out;

				if (chosen_resource[i] != attachment_name_out) {
					auto othfut = Future(rg_frag, chosen_resource[i]);
					rg->attach_in(attachment_name_out, std::move(othfut));
				} else {
					rg->attach_in(attachment_name_out, std::move(rg_frag_fut));
				}
				// hacky way to reference image in the subgraph
				// TODO: a proper way to do this?
				auto si = vuk::make_sampled_image(rg->name.append("::").append(attachment_name_out.to_sv()), imgui_data.font_sci);
				ImGui::Image(&*sampled_images.emplace(si), ImGui::GetContentRegionAvail());
				ImGui::End();
				i++;
			}

			ImGui::Render();
			rg->clear_image("SWAPCHAIN", "SWAPCHAIN+", vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
			rg->attach_swapchain("SWAPCHAIN", swapchain);
			auto fut = util::ImGui_ImplVuk_Render(frame_allocator, Future{ rg, "SWAPCHAIN+" }, imgui_data, ImGui::GetDrawData(), sampled_images);
			auto ptr = fut.get_render_graph();
			auto erg = *compiler.link(std::span{ &ptr, 1 }, {});
			bundle = *acquire_one(*context, swapchain, (*present_ready)[context->get_frame_count() % 3], (*render_complete)[context->get_frame_count() % 3]);
			auto result = *execute_submit(frame_allocator, std::move(erg), std::move(bundle));
			present_to_one(*context, std::move(result));
			sampled_images.clear();
		}
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
