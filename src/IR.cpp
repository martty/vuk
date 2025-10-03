#include <vuk/IR.hpp>

namespace vuk {
	thread_local static std::shared_ptr<IRModule> current_module = std::make_shared<IRModule>();

	std::shared_ptr<IRModule>& get_current_module() {
		return current_module;
	}

	void set_current_module(std::shared_ptr<IRModule> module) {
		current_module = std::move(module);
	}
} // namespace vuk
