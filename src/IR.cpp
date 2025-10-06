#include <vuk/ir/IR.hpp>

namespace vuk {
	thread_local std::shared_ptr<IRModule> current_module = std::make_shared<IRModule>();
} // namespace vuk
