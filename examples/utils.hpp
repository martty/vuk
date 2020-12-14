#pragma once

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <vector>
#include <utility>
#include <VkBootstrap.h>
#include <imgui.h>
#include "vuk/Types.hpp"
#include "vuk/Context.hpp"
#include "vuk/Swapchain.hpp"
#include <string_view>
#include <fstream>
#include <sstream>

namespace vuk {
	class PerThreadContext;
	struct Pass;
	using Name = std::string_view;
	struct RenderGraph;
}

namespace util {
	struct Vertex {
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec3 tangent;
		glm::vec3 bitangent;
		glm::vec2 uv_coordinates;
	};

	using Mesh = std::pair<std::vector<Vertex>, std::vector<unsigned>>;

	inline Mesh generate_cube() {
		// clang-format off
		return Mesh(std::vector<Vertex> {
			// back
			Vertex{ {-1, -1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {1, 1} }, Vertex{ {1, 1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, 0} },
				Vertex{ {1, -1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, 1} }, Vertex{ {1, 1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, 0} },
				Vertex{ {-1, -1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {1, 1} }, Vertex{ {-1, 1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {1, 0} },
				// front 
				Vertex{ {-1, -1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {0, 1} }, Vertex{ {1, -1, 1}, {0, 0, 1},{1, 0.0, 0}, {0, 1, 0}, {1, 1} },
				Vertex{ {1, 1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {1, 0} }, Vertex{ {1, 1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {1, 0} },
				Vertex{ {-1, 1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {0, 0} }, Vertex{ {-1, -1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {0, 1} },
				// left 
				Vertex{ {-1, 1, -1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 0} }, Vertex{ {-1, -1, -1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1} },
				Vertex{ {-1, 1, 1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0} }, Vertex{ {-1, -1, -1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1} },
				Vertex{ {-1, -1, 1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 1} }, Vertex{ {-1, 1, 1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0} },
				// right 
				Vertex{ {1, 1, 1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {0, 0} }, Vertex{ {1, -1, -1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {1, 1} },
				Vertex{ {1, 1, -1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {1, 0} }, Vertex{ {1, -1, -1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {1, 1} },
				Vertex{ {1, 1, 1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {0, 0} }, Vertex{ {1, -1, 1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {0, 1} },
				// bottom 
				Vertex{ {-1, -1, -1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1} }, Vertex{ {1, -1, -1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {1, 1} },
				Vertex{ {1, -1, 1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {1, 0} }, Vertex{ {1, -1, 1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {1, 0} },
				Vertex{ {-1, -1, 1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0} }, Vertex{ {-1, -1, -1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1} },
				// top 
				Vertex{ {-1, 1, -1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0} }, Vertex{ {1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {1, 1} },
				Vertex{ {1, 1, -1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {1, 0} }, Vertex{ {1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {1, 1} },
				Vertex{ {-1, 1, -1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0} }, Vertex{ {-1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0, 1} } },
			{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 });
		// clang-format on
	}

	inline vuk::Swapchain make_swapchain(vkb::Device vkbdevice) {
		vkb::SwapchainBuilder swb(vkbdevice);
		swb.set_desired_format(vuk::SurfaceFormatKHR{ vuk::Format::eR8G8B8A8Srgb, vuk::ColorSpaceKHR::eSrgbNonlinear });
		swb.add_fallback_format(vuk::SurfaceFormatKHR{ vuk::Format::eB8G8R8A8Srgb, vuk::ColorSpaceKHR::eSrgbNonlinear });
		swb.set_desired_present_mode((VkPresentModeKHR)vuk::PresentModeKHR::eImmediate);
		swb.set_image_usage_flags(VkImageUsageFlagBits::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		auto vkswapchain = swb.build();

		vuk::Swapchain sw;
		auto images = vkswapchain->get_images();
		auto views = vkswapchain->get_image_views();

		for (auto& i : *images) {
			sw.images.push_back(i);
		}
		for (auto& i : *views) {
			sw.image_views.emplace_back();
			sw.image_views.back().payload = i;
		}
		sw.extent = vuk::Extent2D{ vkswapchain->extent.width, vkswapchain->extent.height };
		sw.format = vuk::Format(vkswapchain->image_format);
		sw.surface = vkbdevice.surface;
		sw.swapchain = vkswapchain->swapchain;
		return sw;
	}

	struct ImGuiData {
		vuk::Texture font_texture;
		vuk::SamplerCreateInfo font_sci;
		std::unique_ptr<vuk::SampledImage> font_si;
	};
	ImGuiData ImGui_ImplVuk_Init(vuk::PerThreadContext& ptc);
	void ImGui_ImplVuk_Render(vuk::PerThreadContext& ptc, vuk::RenderGraph& rg, vuk::Name src_target, vuk::Name use_target, ImGuiData& data, ImDrawData* draw_data);

	inline std::string read_entire_file(const std::string& path) {
		std::ostringstream buf;
		std::ifstream input(path.c_str());
		assert(input);
		buf << input.rdbuf();
		return buf.str();
	}
}