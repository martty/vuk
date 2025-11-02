#include <vuk/ir/IR.hpp>

namespace vuk {
	thread_local std::shared_ptr<IRModule> current_module = std::make_shared<IRModule>();
} // namespace vuk

namespace std {
	size_t hash<vuk::Ref>::operator()(vuk::Ref const& x) const noexcept {
		size_t h = 0;
		hash_combine(h, x.node, x.index);
		return h;
	}
} // namespace std
