#pragma once

#include <string_view>

namespace vuk {
	class Context;
	class InflightContext;
	class PerThreadContext;
	
	using Name = std::string_view;

	class CommandBuffer;

	struct Swapchain;
	using SwapChainRef = Swapchain *;
}
