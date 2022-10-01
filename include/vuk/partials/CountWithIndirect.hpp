#pragma once

#include "vuk/vuk_fwd.hpp"

namespace vuk {

	struct CountWithIndirect {
		CountWithIndirect(uint32_t count, uint32_t wg_size) : workgroup_count((uint32_t)idivceil(count, wg_size)), count(count) {}

		uint32_t workgroup_count;
		uint32_t yz[2] = { 1, 1 };
		uint32_t count;
	};

} // namespace vuk