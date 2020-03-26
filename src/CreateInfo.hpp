#pragma once

namespace vuk {
	template<class T>
	struct create_info;

	template<class T>
	using create_info_t = typename create_info<T>::type;
}
