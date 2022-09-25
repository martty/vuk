#pragma once

#include <string_view>

#include "vuk/Image.hpp"

namespace vuk {
	std::string_view image_view_type_to_sv(ImageViewType) noexcept;
}