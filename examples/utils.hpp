#pragma once

#include <VkBootstrap.h>
#include <fstream>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <imgui.h>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>
#include <vuk/Context.hpp>
#include <vuk/Future.hpp>
#include <vuk/Swapchain.hpp>
#include <vuk/Types.hpp>
#include <vuk/vuk_fwd.hpp>

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

	inline vuk::Swapchain make_swapchain(vuk::Allocator allocator, vkb::Device vkbdevice, VkSurfaceKHR surface, std::optional<vuk::Swapchain> old_swapchain) {
		vkb::SwapchainBuilder swb(vkbdevice, surface);
		swb.set_desired_format(vuk::SurfaceFormatKHR{ vuk::Format::eR8G8B8A8Srgb, vuk::ColorSpaceKHR::eSrgbNonlinear });
		swb.add_fallback_format(vuk::SurfaceFormatKHR{ vuk::Format::eB8G8R8A8Srgb, vuk::ColorSpaceKHR::eSrgbNonlinear });
		swb.set_desired_present_mode((VkPresentModeKHR)vuk::PresentModeKHR::eImmediate);
		swb.set_image_usage_flags(VkImageUsageFlagBits::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VkImageUsageFlagBits::VK_IMAGE_USAGE_TRANSFER_DST_BIT);

		bool is_recycle = false;
		vkb::Result<vkb::Swapchain> vkswapchain = { vkb::Swapchain{} };
		if (!old_swapchain) {
			vkswapchain = swb.build();
			old_swapchain.emplace(allocator, vkswapchain->image_count);
		} else {
			is_recycle = true;
			swb.set_old_swapchain(old_swapchain->swapchain);
			vkswapchain = swb.build();
		}

		if (is_recycle) {
			allocator.deallocate(std::span{ &old_swapchain->swapchain, 1 });
			for (auto& iv : old_swapchain->images) {
				allocator.deallocate(std::span{ &iv.image_view, 1 });
			}
		}

		auto images = *vkswapchain->get_images();
		auto views = *vkswapchain->get_image_views();

		old_swapchain->images.clear();

		for (auto i = 0; i < images.size(); i++) {
			vuk::ImageAttachment ia;
			ia.extent = { vkswapchain->extent.width, vkswapchain->extent.height, 1 };
			ia.format = (vuk::Format)vkswapchain->image_format;
			ia.image = vuk::Image{ images[i], nullptr };
			ia.image_view = vuk::ImageView{ { 0 }, views[i] };
			ia.view_type = vuk::ImageViewType::e2D;
			ia.sample_count = vuk::Samples::e1;
			ia.base_level = ia.base_layer = 0;
			ia.level_count = ia.layer_count = 1;
			old_swapchain->images.push_back(ia);
		}

		old_swapchain->swapchain = vkswapchain->swapchain;
		old_swapchain->surface = surface;
		return std::move(*old_swapchain);
	}

	struct ImGuiData {
		vuk::Unique<vuk::Image> font_image;
		vuk::Unique<vuk::ImageView> font_image_view;
		vuk::SamplerCreateInfo font_sci;
		vuk::ImageAttachment font_ia;
	};
	ImGuiData ImGui_ImplVuk_Init(vuk::Allocator& allocator);
	vuk::Value<vuk::ImageAttachment> ImGui_ImplVuk_Render(vuk::Allocator& allocator,
	                                                      vuk::Value<vuk::ImageAttachment> target,
	                                                      ImGuiData& data,
	                                                      ImDrawData* draw_data,
	                                                      std::vector<vuk::Value<vuk::ImageAttachment>>& sampled_images);

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