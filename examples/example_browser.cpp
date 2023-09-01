#include "../src/RenderGraphUtil.hpp"
#include "example_runner.hpp"
#include "vuk/RenderGraphReflection.hpp"

std::vector<vuk::QualifiedName> chosen_resource;

bool render_all = true;
vuk::SingleSwapchainRenderBundle bundle;

void vuk::ExampleRunner::render() {
	Compiler compiler;
	chosen_resource.resize(examples.size());

	vuk::wait_for_futures_explicit(*superframe_allocator, compiler, futures);
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

		auto& frame_resource = superframe_resource->get_next_frame();
		context->next_frame();

		Allocator frame_allocator(frame_resource);
		if (!render_all) { // render a single full window example
			RenderGraph rg("runner");
			vuk::Name attachment_name = item_current->name;
			rg.attach_swapchain("_swp", swapchain);
			rg.clear_image("_swp", attachment_name, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
			auto fut = item_current->render(*this, frame_allocator, Future{ std::make_shared<RenderGraph>(std::move(rg)), attachment_name });
			ImGui::Render();

			fut = util::ImGui_ImplVuk_Render(frame_allocator, std::move(fut), imgui_data, ImGui::GetDrawData(), sampled_images);
			// make a new RG that will take care of putting the swapchain image into present and releasing it from the rg
			std::shared_ptr<RenderGraph> rg_p(std::make_shared<RenderGraph>("presenter"));
			rg_p->attach_in("_src", std::move(fut));
			// we tell the rendergraph that _src will be used for presenting after the rendergraph
			rg_p->release_for_present("_src");
			auto erg = *compiler.link(std::span{ &rg_p, 1 }, {});
			bundle = *acquire_one(*context, swapchain, (*present_ready)[context->get_frame_count() % 3], (*render_complete)[context->get_frame_count() % 3]);
			auto result = *execute_submit(frame_allocator, std::move(erg), std::move(bundle));
			present_to_one(*context, std::move(result));
			sampled_images.clear();
		} else { // render all examples as imgui windows
			std::shared_ptr<RenderGraph> rg = std::make_shared<RenderGraph>("runner");

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
				Name attachment_name_out = Name(std::string(ex->name) + "_final");
				auto rg_frag = rg_frag_fut.get_render_graph();
				compiler.compile({ &rg_frag, 1 }, {});
				if (auto use_chains = compiler.get_use_chains(); use_chains.size() > 1) {
					const auto& bound_attachments = compiler.get_bound_attachments();
					for (const auto head : use_chains) {
						if (head->type != Resource::Type::eImage) {
							continue;
						}

						auto& att_info = compiler.get_chain_attachment(head);
						auto samples = att_info.attachment.sample_count.count;
						bool disable = (samples != vuk::SampleCountFlagBits::eInfer && samples != vuk::SampleCountFlagBits::e1);

						auto maybe_name = compiler.get_last_use_name(head);
						
						std::string btn_id = "";
						if (maybe_name->name.to_sv() == attachment_name_out) {
							btn_id = "F";
						} else {
							auto usage = compiler.compute_usage(head);
							if (usage & vuk::ImageUsageFlagBits::eColorAttachment) {
								btn_id += "C";
							} else if (usage & vuk::ImageUsageFlagBits::eDepthStencilAttachment) {
								btn_id += "D";
							} else if (usage & (vuk::ImageUsageFlagBits::eTransferSrc | vuk::ImageUsageFlagBits::eTransferDst)) {
								btn_id += "X";
							}
						}
						if (disable) {
							btn_id += " (MS)";
						} else {
							btn_id += "##" + std::string(att_info.name.prefix.to_sv()) + std::string(att_info.name.name.to_sv());
						}
						if (disable) {
							ImGui::TextDisabled("%s", btn_id.c_str());
						} else {
							if (maybe_name) {
								if (ImGui::Button(btn_id.c_str())) {
									if (maybe_name->name.to_sv() == ex->name) {
										chosen_resource[i] = QualifiedName{ {}, attachment_name_out };
									} else {
										if (maybe_name) {
											chosen_resource[i] = *maybe_name;
										}
									}
								}
							}
						}
						if (ImGui::IsItemHovered())
							ImGui::SetTooltip("%s", att_info.name.name.c_str());
						ImGui::SameLine();
					}
					ImGui::NewLine();
				}
				if (chosen_resource[i].is_invalid())
					chosen_resource[i].name = attachment_name_out;

				if (chosen_resource[i].name != attachment_name_out) {
					auto othfut = Future(rg_frag, chosen_resource[i]);
					rg->attach_in(attachment_name_out, std::move(othfut));
				} else {
					rg->attach_in(attachment_name_out, std::move(rg_frag_fut));
				}
				auto si = vuk::make_sampled_image(NameReference{ rg.get(), QualifiedName({}, attachment_name_out) }, imgui_data.font_sci);
				ImGui::Image(&*sampled_images.emplace(si), ImGui::GetContentRegionAvail());
				ImGui::End();
				i++;
			}

			ImGui::Render();
			rg->clear_image("SWAPCHAIN", "SWAPCHAIN+", vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
			rg->attach_swapchain("SWAPCHAIN", swapchain);
			auto fut = util::ImGui_ImplVuk_Render(frame_allocator, Future{ rg, "SWAPCHAIN+" }, imgui_data, ImGui::GetDrawData(), sampled_images);
			std::shared_ptr<RenderGraph> rg_p(std::make_shared<RenderGraph>("presenter"));
			rg_p->attach_in("_src", std::move(fut));
			// we tell the rendergraph that _src will be used for presenting after the rendergraph
			rg_p->release_for_present("_src");
			auto erg = *compiler.link(std::span{ &rg_p, 1 }, {});
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
