#pragma once

#include <VkBootstrap.h>
#include <fstream>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <imgui.h>
#include <plf_colony.h>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>
#include <vuk/Context.hpp>
#include <vuk/Swapchain.hpp>
#include <vuk/Types.hpp>
#include <vuk/vuk_fwd.hpp>

namespace vuk {
	struct RenderGraph;
} // namespace vuk

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

	inline vuk::Swapchain make_swapchain(vkb::Device vkbdevice, std::optional<VkSwapchainKHR> old_swapchain) {
		vkb::SwapchainBuilder swb(vkbdevice);
		swb.set_desired_format(vuk::SurfaceFormatKHR{ vuk::Format::eR8G8B8A8Srgb, vuk::ColorSpaceKHR::eSrgbNonlinear });
		swb.add_fallback_format(vuk::SurfaceFormatKHR{ vuk::Format::eB8G8R8A8Srgb, vuk::ColorSpaceKHR::eSrgbNonlinear });
		swb.set_desired_present_mode((VkPresentModeKHR)vuk::PresentModeKHR::eImmediate);
		swb.set_image_usage_flags(VkImageUsageFlagBits::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT);
		if (old_swapchain) {
			swb.set_old_swapchain(*old_swapchain);
		}
		auto vkswapchain = swb.build();

		vuk::Swapchain sw{};
		auto images = *vkswapchain->get_images();
		auto views = *vkswapchain->get_image_views();

		for (auto& i : images) {
			sw.images.push_back(vuk::Image{ i, nullptr });
		}
		for (auto& i : views) {
			sw.image_views.emplace_back();
			sw.image_views.back().payload = i;
			sw.image_views.back().id = 0;
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
	ImGuiData ImGui_ImplVuk_Init(vuk::Allocator& allocator);
	vuk::Future ImGui_ImplVuk_Render(vuk::Allocator& allocator,
	                                 vuk::Future target,
	                                 ImGuiData& data,
	                                 ImDrawData* draw_data,
	                                 const plf::colony<vuk::SampledImage>& sampled_images);

	inline std::string read_entire_file(const std::string& path) {
		std::ostringstream buf;
		std::ifstream input(path.c_str());
		assert(input);
		buf << input.rdbuf();
		return buf.str();
	}

	inline std::vector<uint32_t> read_spirv(const std::string& path) {
		std::ostringstream buf;
		std::ifstream input(path.c_str(), std::ios::ate | std::ios::binary);
		assert(input);
		size_t file_size = (size_t)input.tellg();
		std::vector<uint32_t> buffer(file_size / sizeof(uint32_t));
		input.seekg(0);
		input.read(reinterpret_cast<char*>(buffer.data()), file_size);
		return buffer;
	}
} // namespace util