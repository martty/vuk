#include <vuk/ir/IR.hpp>
#include <vuk/ir/IRPass.hpp>

namespace vuk {
	thread_local std::shared_ptr<IRModule> current_module = std::make_shared<IRModule>();

	struct AllocaCtx : IREvalContext {
		std::vector<void*> allocated;

		void* allocate_host_memory(size_t size) override {
			void* ptr = malloc(size);
			allocated.push_back(ptr);
			return ptr;
		}

		~AllocaCtx() {
			for (auto ptr : allocated) {
				free(ptr);
			}
		}
	};

	Result<void*, CannotBeConstantEvaluated> eval(Ref ref) {
		AllocaCtx ctx;
		return ctx.eval(ref);
	}
} // namespace vuk

namespace std {
	size_t hash<vuk::Ref>::operator()(vuk::Ref const& x) const noexcept {
		size_t h = 0;
		hash_combine(h, x.node, x.index);
		return h;
	}
} // namespace std
