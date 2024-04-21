#pragma once

#include "vuk/runtime/vk/Allocator.hpp"
#include "vuk/Config.hpp"
#include "vuk/Types.hpp"
#include "vuk/vuk_fwd.hpp"

#include <vector>

namespace vuk {
	enum class ColorSpaceKHR {
		eSrgbNonlinear = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
		eDisplayP3NonlinearEXT = VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT,
		eExtendedSrgbLinearEXT = VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT,
		eDisplayP3LinearEXT = VK_COLOR_SPACE_DISPLAY_P3_LINEAR_EXT,
		eDciP3NonlinearEXT = VK_COLOR_SPACE_DCI_P3_NONLINEAR_EXT,
		eBt709LinearEXT = VK_COLOR_SPACE_BT709_LINEAR_EXT,
		eBt709NonlinearEXT = VK_COLOR_SPACE_BT709_NONLINEAR_EXT,
		eBt2020LinearEXT = VK_COLOR_SPACE_BT2020_LINEAR_EXT,
		eHdr10St2084EXT = VK_COLOR_SPACE_HDR10_ST2084_EXT,
		eDolbyvisionEXT = VK_COLOR_SPACE_DOLBYVISION_EXT,
		eHdr10HlgEXT = VK_COLOR_SPACE_HDR10_HLG_EXT,
		eAdobergbLinearEXT = VK_COLOR_SPACE_ADOBERGB_LINEAR_EXT,
		eAdobergbNonlinearEXT = VK_COLOR_SPACE_ADOBERGB_NONLINEAR_EXT,
		ePassThroughEXT = VK_COLOR_SPACE_PASS_THROUGH_EXT,
		eExtendedSrgbNonlinearEXT = VK_COLOR_SPACE_EXTENDED_SRGB_NONLINEAR_EXT,
		eDisplayNativeAMD = VK_COLOR_SPACE_DISPLAY_NATIVE_AMD,
		eVkColorspaceSrgbNonlinear = VK_COLORSPACE_SRGB_NONLINEAR_KHR,
		eDciP3LinearEXT = VK_COLOR_SPACE_DCI_P3_LINEAR_EXT
	};

	struct SurfaceFormatKHR {
		Format format = Format::eUndefined;
		ColorSpaceKHR colorSpace = ColorSpaceKHR::eSrgbNonlinear;

		operator VkSurfaceFormatKHR const&() const noexcept {
			return *reinterpret_cast<const VkSurfaceFormatKHR*>(this);
		}

		operator VkSurfaceFormatKHR&() noexcept {
			return *reinterpret_cast<VkSurfaceFormatKHR*>(this);
		}
		bool operator==(SurfaceFormatKHR const& rhs) const noexcept {
			return (format == rhs.format) && (colorSpace == rhs.colorSpace);
		}

		bool operator!=(SurfaceFormatKHR const& rhs) const noexcept {
			return !operator==(rhs);
		}
	};
	static_assert(sizeof(SurfaceFormatKHR) == sizeof(VkSurfaceFormatKHR), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<SurfaceFormatKHR>::value, "struct wrapper is not a standard layout!");

	enum class PresentModeKHR {
		eImmediate = VK_PRESENT_MODE_IMMEDIATE_KHR,
		eMailbox = VK_PRESENT_MODE_MAILBOX_KHR,
		eFifo = VK_PRESENT_MODE_FIFO_KHR,
		eFifoRelaxed = VK_PRESENT_MODE_FIFO_RELAXED_KHR,
		eSharedDemandRefresh = VK_PRESENT_MODE_SHARED_DEMAND_REFRESH_KHR,
		eSharedContinuousRefresh = VK_PRESENT_MODE_SHARED_CONTINUOUS_REFRESH_KHR
	};

	struct ImageAttachment;

	struct Swapchain {
		Swapchain(Allocator allocator, size_t image_count);
		Swapchain(const Swapchain&) = delete;
		Swapchain(Swapchain&&) noexcept;

		Swapchain& operator=(const Swapchain&) = delete;
		Swapchain& operator=(Swapchain&&) noexcept;

		~Swapchain();

		Allocator allocator;
		VkSwapchainKHR swapchain = VK_NULL_HANDLE;
		VkSurfaceKHR surface = VK_NULL_HANDLE;

		std::vector<ImageAttachment> images;
		uint32_t linear_index = 0;
		uint32_t image_index;
		std::vector<VkSemaphore> semaphores; /* present_rdy_0 render_complete_0 present_rdy_1 render_complete_1 ... */
		VkResult acquire_result;
	};
} // namespace vuk