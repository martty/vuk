#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/Types.hpp"

#include <deque>
#include <functional>
#include <span>
#include <vector>

namespace vuk {
	struct SyncPoint {
		DomainFlagBits domain = DomainFlagBits::eNone; // domain of the point
		uint64_t visibility;                           // results are available if waiting for {domain, visibility}
	};

	/// @brief Encapsulates a SyncPoint that can be synchronized against in the future
	struct Signal {
	public:
		enum class Status {
			eDisarmed,       // the Signal is in the initial state - it must be armed before it can be sync'ed against
			eSynchronizable, // this syncpoint has been submitted (result is available on device with appropriate sync)
			eHostAvailable   // the result is available on host, available on device without sync
		};

		Status status = Status::eDisarmed;
		SyncPoint source;
	};

	struct AcquireRelease : Signal {
		Access last_use; // last access performed on resource before signalling
	};

	struct TypeDebugInfo {
		std::string name;
	};

	struct Type {
		enum TypeKind { MEMORY_TY, INTEGER_TY, IMAGE_TY, BUFFER_TY, IMBUED_TY, ALIASED_TY, OPAQUE_FN_TY } kind;

		TypeDebugInfo* debug_info = nullptr;

		union {
			struct {
				size_t width;
			} integer;
			struct {
				Type* T;
				Access access;
			} imbued;
			struct {
				Type* T;
				size_t ref_idx;
			} aliased;
			struct {
				std::span<Type* const> args;
				std::span<Type* const> return_types;
				int execute_on;
				std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>)>* callback;
			} opaque_fn;
		};

		static Type* stripped(Type* t) {
			switch (t->kind) {
			case IMBUED_TY:
				return stripped(t->imbued.T);
			case ALIASED_TY:
				return stripped(t->aliased.T);
			default:
				return t;
			}
		}

		bool is_image() {
			return kind == IMAGE_TY;
		}

		bool is_buffer() {
			return kind == BUFFER_TY;
		}

		~Type() {}
	};

	struct RG;

	struct Node;

	struct Ref {
		Node* node = nullptr;
		size_t index;

		Type* type() const;

		explicit constexpr operator bool() const noexcept {
			return node != nullptr;
		}

		constexpr std::strong_ordering operator<=>(const Ref&) const noexcept = default;
	};

	struct SchedulingInfo {
		DomainFlags required_domain;
	};

	struct NodeDebugInfo {
		std::vector<std::string> result_names;
	};

	struct Node {
		enum Kind { NOP, PLACEHOLDER, CONSTANT, VALLOC, IMPORT, CALL, CLEAR, DIVERGE, CONVERGE, RESOLVE, SIGNAL, WAIT, ACQUIRE, RELEASE } kind;
		std::span<Type* const> type;
		NodeDebugInfo* debug_info = nullptr;
		SchedulingInfo* scheduling_info = nullptr;
		union {
			struct {
			} placeholder;
			struct {
				void* value;
			} constant;
			struct {
				std::span<Ref> args;
				std::optional<Allocator> allocator;
			} valloc;
			struct {
				void* value;
			} import;
			struct {
				Ref fn;
				std::span<Ref> args;
			} call;
			struct {
				const Ref dst;
				Clear* cv;
			} clear;
			struct {
				const Ref initial;
				Subrange::Image subrange;
			} diverge;
			struct {
				std::span<Ref> diverged;
			} converge;
			struct {
				const Ref source_ms;
				const Ref source_ss;
				const Ref dst_ss;
			} resolve;
			struct {
				const Ref src;
				Signal* signal;
			} signal;
			struct {
				const Ref dst;
				Signal* signal;
			} wait;
			struct {
				const Ref dst;
				AcquireRelease* acquire;
			} acquire;
			struct {
				Ref src;
				AcquireRelease* release;
			} release;
		};

		std::string_view kind_to_sv() {
			switch (kind) {
			case VALLOC:
				return "valloc";
			case CALL:
				return "call";
			}
			assert(0);
			return "";
		}
	};

	inline Ref first(Node* node) {
		assert(node->type.size() > 0);
		return { node, 0 };
	}

	inline Ref nth(Node* node, size_t idx) {
		assert(node->type.size() > idx);
		return { node, idx };
	}

	inline Type* Ref::type() const {
		return node->type[index];
	}

	struct RG {
		RG() {
			builtin_image = &types.emplace_back(Type{ .kind = Type::IMAGE_TY });
			builtin_buffer = &types.emplace_back(Type{ .kind = Type::BUFFER_TY });
		}

		std::deque<Node> op_arena;
		char* debug_arena;

		std::deque<Type> types;
		Type* builtin_image;
		Type* builtin_buffer;

		std::vector<std::shared_ptr<RG>> subgraphs;
		// uint64_t current_hash = 0;

		void* ensure_space(size_t size) {
			return &op_arena.emplace_back(Node{});
		}

		Node* emplace_op(Node v, NodeDebugInfo = {}) {
			return new (ensure_space(sizeof(Node))) Node(v);
		}

		Type* emplace_type(Type&& t, TypeDebugInfo = {}) {
			return &types.emplace_back(std::move(t));
		}

		void reference_RG(std::shared_ptr<RG> other) {
			subgraphs.emplace_back(other);
		}

		// TYPES
		Type* make_imbued_ty(Type* ty, Access access) {
			return emplace_type(Type{ .kind = Type::IMBUED_TY, .imbued = { .T = ty, .access = access } });
		}

		Type* make_aliased_ty(Type* ty, size_t ref_idx) {
			return emplace_type(Type{ .kind = Type::ALIASED_TY, .aliased = { .T = ty, .ref_idx = ref_idx } });
		}

		Type* u64() {
			return emplace_type(Type{ .kind = Type::INTEGER_TY, .integer = { .width = 64 } });
		}

		// OPS

		void name_outputs(Node* node, std::vector<std::string> names) {
			if (!node->debug_info) {
				node->debug_info = new NodeDebugInfo;
			}
			node->debug_info->result_names.assign(names.begin(), names.end());
		}

		void name_output(Ref ref, std::string name) {
			if (!ref.node->debug_info) {
				ref.node->debug_info = new NodeDebugInfo;
			}
			auto& names = ref.node->debug_info->result_names;
			if (names.size() <= ref.index) {
				names.resize(ref.index + 1);
			}
			names[ref.index] = name;
		}

		Ref make_declare_image(ImageAttachment value) {
			assert(0);
			return first(emplace_op(Node{ .kind = Node::VALLOC, .type = std::span{ &builtin_image, 1 }, .valloc = {} }));
		}

		Ref make_declare_buffer(Buffer value) {
			auto buf_ptr = new Buffer(value); /* size rest */
			auto args_ptr = new Ref[2];
			auto mem_ty = emplace_type(Type{ .kind = Type::MEMORY_TY });
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ &mem_ty, 1 }, .constant = { .value = buf_ptr } }));
			auto u64_ty = u64();
			if (value.size > 0) {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ &u64_ty, 1 }, .constant = { .value = &buf_ptr->size } }));
			} else {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ &u64_ty, 1 } }));
			}

			return first(emplace_op(Node{ .kind = Node::VALLOC, .type = std::span{ &builtin_buffer, 1 }, .valloc = { .args = std::span(args_ptr, 2) } }));
		}

		Ref make_clear_image(Ref dst, Clear cv) {
			return first(emplace_op(Node{ .kind = Node::CLEAR, .type = std::span{ &builtin_image, 1 }, .clear = { .dst = dst, .cv = new Clear(cv) } }));
		}

		Type* make_opaque_fn_ty(std::span<Type* const> args,
		                        std::span<Type* const> ret_types,
		                        DomainFlags execute_on,
		                        std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>)> callback) {
			auto arg_ptr = new Type*[args.size()];
			std::copy(args.begin(), args.end(), arg_ptr);
			auto ret_ty_ptr = new Type*[ret_types.size()];
			std::copy(ret_types.begin(), ret_types.end(), ret_ty_ptr);
			return emplace_type(Type{ .kind = Type::OPAQUE_FN_TY,
			                          .opaque_fn = { .args = std::span(arg_ptr, args.size()),
			                                         .return_types = std::span(ret_ty_ptr, ret_types.size()),
			                                         .execute_on = execute_on.m_mask,
			                                         .callback = new std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>)>(callback) } });
		}

		Ref make_declare_fn(Type* const fn_ty) {
			auto ty = new Type*(fn_ty);
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { nullptr } }));
		}

		template<class... Refs>
		Node* make_call(Ref fn, Refs... args) {
			Ref* args_ptr = new Ref[sizeof...(args)]{ args... };
			decltype(Node::call) call = { .fn = fn, .args = std::span(args_ptr, sizeof...(args)) };
			Node n{};
			n.kind = Node::CALL;
			n.type = fn.type()->opaque_fn.return_types;
			n.call = call;
			return emplace_op(n);
		}

		Node* make_release(Ref src, AcquireRelease* acq_rel) {
			return emplace_op(Node{ .kind = Node::RELEASE, .release = { .src = src, .release = acq_rel } });
		}
	};
} // namespace vuk