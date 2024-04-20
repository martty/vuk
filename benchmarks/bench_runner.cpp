#include "bench_runner.hpp"
#include "../src/RenderGraphUtil.hpp"

std::vector<std::string> chosen_resource;

vuk::BenchRunner::BenchRunner() {
	vkb::InstanceBuilder builder;
	builder
	    .set_debug_callback([](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	                           VkDebugUtilsMessageTypeFlagsEXT messageType,
	                           const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	                           void* pUserData) -> VkBool32 {
		    auto ms = vkb::to_string_message_severity(messageSeverity);
		    auto mt = vkb::to_string_message_type(messageType);
		    printf("[%s: %s](user defined)\n%s\n", ms, mt, pCallbackData->pMessage);
		    return VK_FALSE;
	    })
	    .set_app_name("vuk_bench")
	    .set_engine_name("vuk")
	    .require_api_version(1, 2, 0)
	    .set_app_version(0, 1, 0);
	auto inst_ret = builder.build();
	if (!inst_ret.has_value()) {
		// error
	}
	vkbinstance = inst_ret.value();
	auto instance = vkbinstance.instance;
	vkb::PhysicalDeviceSelector selector{ vkbinstance };
	window = create_window_glfw("vuk-benchmarker", false);
	surface = create_surface_glfw(vkbinstance.instance, window);
	selector.set_surface(surface).set_minimum_version(1, 0).add_required_extension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
	auto phys_ret = selector.select();
	if (!phys_ret.has_value()) {
		// error
	}
	vkb::PhysicalDevice vkbphysical_device = phys_ret.value();
	physical_device = vkbphysical_device.physical_device;

	vkb::DeviceBuilder device_builder{ vkbphysical_device };
	VkPhysicalDeviceVulkan12Features vk12features{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
	vk12features.timelineSemaphore = true;
	vk12features.descriptorBindingPartiallyBound = true;
	vk12features.descriptorBindingUpdateUnusedWhilePending = true;
	vk12features.shaderSampledImageArrayNonUniformIndexing = true;
	vk12features.runtimeDescriptorArray = true;
	vk12features.descriptorBindingVariableDescriptorCount = true;
	vk12features.hostQueryReset = true;
	VkPhysicalDeviceSynchronization2FeaturesKHR sync_feat{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR, .synchronization2 = true };
	auto dev_ret = device_builder.add_pNext(&vk12features).add_pNext(&sync_feat).build();
	if (!dev_ret.has_value()) {
		// error
	}
	vkbdevice = dev_ret.value();
	graphics_queue = vkbdevice.get_queue(vkb::QueueType::graphics).value();
	auto graphics_queue_family_index = vkbdevice.get_queue_index(vkb::QueueType::graphics).value();
	device = vkbdevice.device;

	runtime.emplace(RuntimeCreateParameters{ instance, device, physical_device, graphics_queue, graphics_queue_family_index });
	const unsigned num_inflight_frames = 3;
	xdev_rf_alloc.emplace(*runtime, num_inflight_frames);
	global.emplace(*xdev_rf_alloc);
	swapchain = runtime->add_swapchain(util::make_swapchain(vkbdevice, {}));
}

constexpr unsigned stage_wait = 0;
constexpr unsigned stage_warmup = 1;
constexpr unsigned stage_variance = 2;
constexpr unsigned stage_live = 3;
constexpr unsigned stage_complete = 4;

void vuk::BenchRunner::render() {
	Compiler compiler;
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		auto& xdev_frame_resource = xdev_rf_alloc->get_next_frame();
		runtime->next_frame();
		Allocator frame_allocator(xdev_frame_resource);
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x - 552.f, 2));
		ImGui::SetNextWindowSize(ImVec2(550, 0));
		ImGui::Begin("Benchmark", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoResize);

		ImGui::Text("%s", bench->name.data());
		ImGui::SameLine();
		if (current_stage == 0 && ImGui::Button("Start")) {
			current_stage = 1;
		}
		ImGui::NewLine();
		ImGui::Separator();
		for (auto i = 0; i < bench->num_cases; i++) {
			auto& bcase = bench->get_case(i);
			if (ImGui::CollapsingHeader(bcase.label.data(), ImGuiTreeNodeFlags_DefaultOpen)) {
				for (auto j = 0; j < bcase.subcases.size(); j++) {
					bool sel = current_case == i && current_subcase == j;
					ImVec2 size = ImVec2(0.f, 0.f);

					uint32_t runs;
					if (current_stage == stage_warmup) {
						runs = 50;
					} else if (current_stage == stage_variance) {
						runs = 50;
					} else if (current_stage == stage_live) {
						runs = bcase.runs_required[j];
					} else {
						runs = num_runs;
					}

					if (sel && current_stage != stage_wait && current_stage != stage_complete) {
						size.x = (float)num_runs / runs * ImGui::GetContentRegionAvail().x;
					}

					ImGui::Selectable(bcase.subcase_labels[j].data(), &sel, sel ? 0 : ImGuiSelectableFlags_Disabled, size);
					ImGui::Indent();

					auto& lsr = bcase.last_stage_ran[j];
					bool w = sel && current_stage == stage_warmup;
					std::string l1 = "Warmup";
					if (lsr > stage_warmup) {
						l1 += " - done";
					} else if (w) {
						l1 += " (" + std::to_string(num_runs) + " / " + std::to_string(runs) + ")";
					}
					ImGui::Selectable(l1.c_str(), &w, w ? 0 : ImGuiSelectableFlags_Disabled);
					w = sel && current_stage == stage_variance;
					std::string l2 = "Variance estimation";
					if (w) {
						l2 = "Estimating variance (" + std::to_string(num_runs) + " / " + std::to_string(runs) + ")";
					} else if (lsr > stage_variance) {
						l2 = "Estimate (mu=" + std::to_string(bcase.est_mean[j] * 1e6) + " us, sigma=" + std::to_string(bcase.est_variance[j] * 1e12) +
						     " us2, runs: " + std::to_string(bcase.runs_required[j]) + ")";
					}
					ImGui::Selectable(l2.c_str(), &w, w ? 0 : ImGuiSelectableFlags_Disabled);
					w = sel && current_stage == stage_live;
					std::string l3 = "Sampling";
					if (w) {
						l3 = "Sampling (" + std::to_string(num_runs) + " / " + std::to_string(runs) + ")";
					} else if (lsr > stage_live) {
						l3 = "Result (mu=" + std::to_string(bcase.mean[j] * 1e6) + " us, sigma=" + std::to_string(bcase.variance[j] * 1e12) +
						     " us2, SEM = " + std::to_string(sqrt(bcase.variance[j] * 1e12 / sqrt(bcase.runs_required[j]))) + " us)";
					}
					ImGui::Selectable(l3.c_str(), &w, w ? 0 : ImGuiSelectableFlags_Disabled);
					if (lsr > stage_live) {
						ImGui::PlotHistogram("Bins", bcase.binned[j].data(), (int)bcase.binned[j].size());
					}
					ImGui::Unindent();
				}
			}
		}

		bench->gui(*this, frame_allocator);

		ImGui::End();

		auto& bench_case = bench->get_case(current_case);
		auto& subcase = bench_case.subcases[current_subcase];
		auto rg = std::make_shared<vuk::RenderGraph>(subcase(*this, frame_allocator, start, end));
		ImGui::Render();

		vuk::Name attachment_name = "_final";
		rg->attach_swapchain("_swp", swapchain);
		rg->clear_image("_swp", attachment_name, vuk::ClearColor{ 0.3f, 0.5f, 0.3f, 1.0f });
		auto fut = util::ImGui_ImplVuk_Render(frame_allocator, Future{ rg, "_final+" }, imgui_data, ImGui::GetDrawData(), sampled_images);
		present(frame_allocator, compiler, swapchain, std::move(fut));
		sampled_images.clear();

		std::optional<double> duration = runtime->retrieve_duration(start, end);
		auto& bcase = bench->get_case(current_case);
		if (!duration) {
			continue;
		} else if (current_stage != stage_complete && current_stage != stage_wait) {
			bcase.timings[current_subcase].push_back(*duration);

			num_runs++;
		}
		// transition between stages
		if (current_stage == stage_warmup && num_runs >= 50) {
			current_stage++;
			bcase.last_stage_ran[current_subcase]++;
			bcase.last_stage_ran[current_subcase]++;

			double& mean = bcase.est_mean[current_subcase];
			mean = 0;
			for (auto& t : bcase.timings[current_subcase]) {
				mean += t;
			}
			num_runs = 0;
			bcase.timings[current_subcase].clear();
		} else if (current_stage == stage_variance && num_runs >= 50) {
			double& mean = bcase.est_mean[current_subcase];
			mean = 0;
			for (auto& t : bcase.timings[current_subcase]) {
				mean += t;
			}
			mean /= num_runs;

			double& variance = bcase.est_variance[current_subcase];
			variance = 0;
			for (auto& t : bcase.timings[current_subcase]) {
				variance += (t - mean) * (t - mean);
			}
			variance *= 1.0 / (num_runs - 1);

			const auto Z = 1.96; // 95% confidence
			bcase.runs_required[current_subcase] = (uint32_t)std::ceil(4 * Z * Z * variance / ((0.1 * mean) * (0.1 * mean)));
			// run at least 128 iterations
			bcase.runs_required[current_subcase] = std::max(bcase.runs_required[current_subcase], 128u);

			current_stage++;
			bcase.last_stage_ran[current_subcase]++;
			// reuse timings for subsequent live
		} else if (current_stage == stage_live && num_runs >= bcase.runs_required[current_subcase]) {
			double& mean = bcase.mean[current_subcase];
			mean = 0;
			double& min = bcase.min_max[current_subcase].first;
			min = DBL_MAX;
			double& max = bcase.min_max[current_subcase].second;
			max = 0;
			for (auto& t : bcase.timings[current_subcase]) {
				mean += t;
				min = std::min(min, t);
				max = std::max(max, t);
			}
			mean /= num_runs;

			auto& bins = bcase.binned[current_subcase];
			bins.resize(64);

			double& variance = bcase.variance[current_subcase];
			variance = 0;
			for (auto& t : bcase.timings[current_subcase]) {
				variance += (t - mean) * (t - mean);
				auto bin_index = (uint32_t)std::floor((bins.size() - 1) * (t - min) / (max - min));
				bins[bin_index]++;
			}
			variance *= 1.0 / (num_runs - 1);

			bcase.last_stage_ran[current_subcase]++;

			//TODO: https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test

			if (bcase.subcases.size() > current_subcase + 1) {
				current_subcase++;
				current_stage = 1;
				num_runs = 0;
				continue;
			}
			if (bench->num_cases > current_case + 1) {
				current_case++;
				current_subcase = 0;
				current_stage = 1;
				num_runs = 0;
				continue;
			}
			current_stage = 0;
			current_case = 0;
			current_subcase = 0;
		}
	}
}

int main() {
	vuk::BenchRunner::get_runner().setup();
	vuk::BenchRunner::get_runner().render();
	vuk::BenchRunner::get_runner().cleanup();
}
