#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/Swapchain.hpp"
#include "vuk/Types.hpp"

#include <deque>
#include <functional>
#include <span>
#include <vector>

namespace vuk {
	struct SyncPoint {
		Executor* executor;
		uint64_t visibility; // results are available if waiting for {executor, visibility}
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
		ResourceUse last_use; // last access performed on resource before signalling
	};

	struct TypeDebugInfo {
		std::string name;
	};

	struct Type {
		enum TypeKind { MEMORY_TY, INTEGER_TY, COMPOSITE_TY, ARRAY_TY, IMBUED_TY, ALIASED_TY, OPAQUE_FN_TY } kind;
		size_t size;

		TypeDebugInfo* debug_info = nullptr;

		union {
			struct {
				uint32_t width;
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
				std::span<Type*> args;
				std::span<Type*> return_types;
				int execute_on;
				std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>, std::span<void*>)>* callback;
			} opaque_fn;
			struct {
				Type* T;
				size_t count;
				size_t stride;
			} array;
			struct {
				std::span<Type*> types;
				std::span<size_t> offsets;
				size_t tag;
			} composite;
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

		static Type* extract(Type* t, size_t index) {
			assert(t->kind == COMPOSITE_TY);
			assert(index < t->composite.types.size());
			return t->composite.types[index];
		}

		static uint32_t hash(Type* t) {
			uint32_t v = 0;
			switch (t->kind) {
			case IMBUED_TY:
				v = Type::hash(t->imbued.T);
				hash_combine_direct(v, (uint32_t)t->imbued.access);
				return v;
			case ALIASED_TY:
				v = Type::hash(t->aliased.T);
				hash_combine_direct(v, (uint32_t)t->aliased.ref_idx);
				return v;
			case MEMORY_TY:
				return 0;
			case INTEGER_TY:
				return t->integer.width;
			case ARRAY_TY:
				v = Type::hash(t->array.T);
				hash_combine_direct(v, (uint32_t)t->array.count);
				return v;
			case COMPOSITE_TY:
				for (int i = 0; i < t->composite.types.size(); i++) {
					hash_combine_direct(v, Type::hash(t->composite.types[i]));
				}
				hash_combine_direct(v, (uint32_t)t->composite.tag);
				return v;
			case OPAQUE_FN_TY:
				hash_combine_direct(v, (uintptr_t)t->opaque_fn.callback >> 32);
				hash_combine_direct(v, (uintptr_t)t->opaque_fn.callback & 0xffffffff);
				return v;
			}
		}

		static std::string to_string(Type* t) {
			switch (t->kind) {
			case IMBUED_TY:
				return to_string(t->imbued.T) + std::string(":") + std::to_string(t->imbued.access);
			case ALIASED_TY:
				return to_string(t->aliased.T) + std::string("@") + std::to_string(t->aliased.ref_idx);
			case MEMORY_TY:
				return "mem";
			case INTEGER_TY:
				return t->integer.width == 32 ? "i32" : "i64";
			case ARRAY_TY:
				return to_string(t->array.T) + "[" + std::to_string(t->array.count) + "]";
			case COMPOSITE_TY:
				if (t->debug_info && !t->debug_info->name.empty()) {
					return t->debug_info->name;
				}
				return "composite:" + std::to_string(t->composite.tag);
			case OPAQUE_FN_TY:
				return "ofn";
			}
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
		SchedulingInfo(DomainFlags required_domains) : required_domains(required_domains) {}
		SchedulingInfo(DomainFlagBits required_domain) : required_domains(required_domain) {}

		DomainFlags required_domains;
	};

	struct NodeDebugInfo {
		std::vector<std::string> result_names;
		std::source_location decl_loc;
	};

	struct Node {
		static constexpr uint8_t MAX_ARGS = 16;

		enum class BinOp { MUL };
		enum Kind {
			NOP,
			PLACEHOLDER,
			CONSTANT,
			CONSTRUCT,
			EXTRACT,
			IMPORT,
			CALL,
			CLEAR,
			DIVERGE,
			CONVERGE,
			RESOLVE,
			SIGNAL,
			WAIT,
			ACQUIRE,
			RELEASE,
			RELACQ, // can realise into ACQUIRE, RELEASE or NOP
			ACQUIRE_NEXT_IMAGE,
			CAST,
			MATH_BINARY
		} kind;
		std::span<Type*> type;
		NodeDebugInfo* debug_info = nullptr;
		SchedulingInfo* scheduling_info = nullptr;

		template<uint8_t c>
		struct Fixed {
			uint8_t arg_count = c;
		};

		struct Variable {
			uint8_t arg_count = (uint8_t)~0u;
		};

		union {
			struct : Fixed<0> {
			} placeholder;
			struct : Fixed<0> {
				void* value;
			} constant;
			struct : Variable {
				std::span<Ref> args;
				std::span<Ref> defs; // for preserving provenance for composite types
				std::optional<Allocator> allocator;
			} construct;
			struct : Fixed<2> {
				Ref composite;
				Ref index;
			} extract;
			struct : Fixed<0> {
				void* value;
			} import;
			struct : Variable {
				std::span<Ref> args;
				Ref fn;
			} call;
			struct : Fixed<1> {
				const Ref dst;
				Clear* cv;
			} clear;
			struct : Fixed<1> {
				const Ref initial;
				Subrange::Image subrange;
			} diverge;
			struct : Variable {
				std::span<Ref> diverged;
			} converge;
			struct : Fixed<3> {
				const Ref source_ms;
				const Ref source_ss;
				const Ref dst_ss;
			} resolve;
			struct : Fixed<1> {
				const Ref src;
				Signal* signal;
			} signal;
			struct : Fixed<1> {
				const Ref dst;
				Signal* signal;
			} wait;
			struct : Fixed<1> {
				Ref arg;
				AcquireRelease* acquire;
			} acquire;
			struct : Fixed<1> {
				Ref src;
				AcquireRelease* release;
				Access dst_access;
				DomainFlagBits dst_domain;
			} release;
			struct : Fixed<1> {
				Ref src;
				AcquireRelease* rel_acq;
				void* value;
			} relacq;
			struct : Fixed<1> {
				Ref swapchain;
			} acquire_next_image;
			struct : Fixed<1> {
				Ref src;
			} cast;
			struct : Fixed<2> {
				Ref a;
				Ref b;
				BinOp op;
			} math_binary;
			struct {
				uint8_t arg_count;
			} generic_node;
			struct {
				uint8_t arg_count;
				Ref args[MAX_ARGS];
			} fixed_node;
			struct {
				uint8_t arg_count;
				std::span<Ref> args;
			} variable_node;
		};

		std::string_view kind_to_sv() {
			switch (kind) {
			case IMPORT:
				return "import";
			case CONSTRUCT:
				return "construct";
			case CALL:
				return "call";
			case EXTRACT:
				return "extract";
			case ACQUIRE:
				return "acquire";
			case RELACQ:
				return "relacq";
			case RELEASE:
				return "release";
			case MATH_BINARY:
				return "math_b";
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

	template<class T>
	T& constant(Ref ref) {
		assert(ref.type()->kind == Type::INTEGER_TY || ref.type()->kind == Type::MEMORY_TY);
		return *reinterpret_cast<T*>(ref.node->constant.value);
	}

	struct CannotBeConstantEvaluated : Exception {
		CannotBeConstantEvaluated(Ref ref) : ref(ref) {}

		Ref ref;

		void throw_this() override {
			throw *this;
		}
	};

	template<class T>
	  requires(std::is_pointer_v<T>)
	T eval(Ref ref) {
		switch (ref.node->kind) {
		case Node::CONSTANT: {
			return static_cast<T>(ref.node->constant.value);
		}
		case Node::CONSTRUCT: {
			return eval<T>(ref.node->construct.args[0]);
		}
		case Node::ACQUIRE: {
			return eval<T>(ref.node->acquire.arg);
		}
		case Node::RELACQ: {
			return eval<T>(ref.node->relacq.src);
		}
		case Node::ACQUIRE_NEXT_IMAGE: {
			Swapchain* swp = eval<Swapchain*>(ref.node->acquire_next_image.swapchain);
			return reinterpret_cast<T>(&swp->images[0]);
		}
		default:
			throw CannotBeConstantEvaluated{ ref };
		}
	}

	template<class T>
	  requires(std::is_arithmetic_v<T>)
	T eval(Ref ref) {
		switch (ref.node->kind) {
		case Node::CONSTANT: {
			return constant<T>(ref);
		}
		case Node::MATH_BINARY: {
			auto& math_binary = ref.node->math_binary;
			switch (math_binary.op) {
			case Node::BinOp::MUL: {
				return eval<T>(math_binary.a) * eval<T>(math_binary.b);
			}
			}
		}
		case Node::EXTRACT: {
			auto composite = ref.node->extract.composite;
			auto index = eval<uint64_t>(ref.node->extract.index);
			if (composite.node->kind == Node::CONSTRUCT) {
				return eval<T>(composite.node->construct.args[index + 1]);
			} else if (composite.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
				auto swp = composite.node->acquire_next_image.swapchain;
				if (swp.node->kind == Node::CONSTRUCT) {
					auto arr = swp.node->construct.args[1]; // array of images
					if (arr.node->kind == Node::CONSTRUCT) {
						auto elem = arr.node->construct.args[1]; // first image
						if (elem.node->kind == Node::CONSTRUCT) {
							return eval<T>(elem.node->construct.args[index + 1]);
						}
					}
				}
			}
			throw CannotBeConstantEvaluated(ref);
		}
		default:
			throw CannotBeConstantEvaluated(ref);
		}
	}

	template<class T>
	  requires(!std::is_pointer_v<T> && !std::is_arithmetic_v<T>)
	T eval(Ref ref) {
		switch (ref.node->kind) {
		case Node::CONSTANT: {
			return constant<T>(ref);
		}
		case Node::EXTRACT: {
			auto composite = ref.node->extract.composite;
			auto index = eval<uint64_t>(ref.node->extract.index);
			if (composite.type()->kind == Type::COMPOSITE_TY) {
				auto offset = composite.type()->composite.offsets[index];
				return *reinterpret_cast<T*>(eval<unsigned char*>(composite) + offset);
			} else if (composite.type()->kind == Type::ARRAY_TY) {
				return eval<T*>(composite)[index];
			} else {
				assert(0);
			}
		}
			/*
			      if (node->kind == Node::CONSTRUCT) {
			  return node->construct.args[0].node->constant.value;
			} else if (node->kind == Node::ACQUIRE_NEXT_IMAGE) {
			  Swapchain* swp = reinterpret_cast<Swapchain*>(get_constant_value(node->acquire_next_image.swapchain.node));
			  return &swp->images[0];
			} else if (node->kind == Node::ACQUIRE) {
			  return node->acquire.arg.node->constant.value;
			} else if (node->kind == Node::RELACQ) {
			  return get_constant_value(node->relacq.src.node);
			} else {
			  assert(0);
			}
			*/
		default:
			throw CannotBeConstantEvaluated(ref);
		}
	}

	struct RG {
		RG() {
			auto mem_ty = emplace_type(Type{ .kind = Type::MEMORY_TY });
			auto image_ = new Type* [9] {
				u32(), u32(), u32(), mem_ty, mem_ty, u32(), u32(), u32(), u32()
			};
			auto image_offsets = new size_t[9]{ offsetof(ImageAttachment, extent) + offsetof(Dimension3D, extent) + offsetof(Extent3D, width),
				                                  offsetof(ImageAttachment, extent) + offsetof(Dimension3D, extent) + offsetof(Extent3D, height),
				                                  offsetof(ImageAttachment, extent) + offsetof(Dimension3D, extent) + offsetof(Extent3D, depth),
				                                  offsetof(ImageAttachment, format),
				                                  offsetof(ImageAttachment, sample_count),
				                                  offsetof(ImageAttachment, base_layer),
				                                  offsetof(ImageAttachment, layer_count),
				                                  offsetof(ImageAttachment, base_level),
				                                  offsetof(ImageAttachment, level_count) };
			builtin_image = emplace_type(Type{ .kind = Type::COMPOSITE_TY,
			                                   .size = sizeof(ImageAttachment),
			                                   .debug_info = new TypeDebugInfo{ "image" },
			                                   .composite = { .types = { image_, 9 }, .offsets = { image_offsets, 9 }, .tag = 0 } });
			auto buffer_ = new Type* [1] {
				u32()
			};
			auto buffer_offsets = new size_t[1]{ offsetof(Buffer, size) };
			builtin_buffer = emplace_type(Type{ .kind = Type::COMPOSITE_TY,
			                                    .size = sizeof(Buffer),
			                                    .debug_info = new TypeDebugInfo{ "buffer" },
			                                    .composite = { .types = { buffer_, 1 }, .offsets = { buffer_offsets, 1 }, .tag = 1 } });
			auto arr_ty = emplace_type(
			    Type{ .kind = Type::ARRAY_TY, .size = 16 * builtin_image->size, .array = { .T = builtin_image, .count = 16, .stride = builtin_image->size } });
			auto swp_ = new Type* [1] {
				arr_ty
			};
			builtin_swapchain = emplace_type(Type{ .kind = Type::COMPOSITE_TY,
			                                       .size = sizeof(Swapchain),
			                                       .debug_info = new TypeDebugInfo{ "swapchain" },
			                                       .composite = { .types = { swp_, 1 }, .tag = 2 } });
		}

		~RG() {}

		std::deque<Node> op_arena;
		char* debug_arena;

		std::deque<Type> types;
		Type* builtin_image;
		Type* builtin_buffer;
		Type* builtin_swapchain;

		std::vector<std::shared_ptr<RG>> subgraphs;
		// uint64_t current_hash = 0;

		void* ensure_space(size_t size) {
			return &op_arena.emplace_back(Node{});
		}

		Node* emplace_op(Node v, NodeDebugInfo = {}) {
			return new (ensure_space(sizeof(Node))) Node(v);
		}

		Type* emplace_type(Type t, TypeDebugInfo = {}) {
			return &types.emplace_back(std::move(t));
		}

		void reference_RG(std::shared_ptr<RG> other) {
			subgraphs.emplace_back(other);
		}

		void name_outputs(Node* node, std::vector<std::string> names) {
			if (!node->debug_info) {
				node->debug_info = new NodeDebugInfo;
			}
			node->debug_info->result_names.assign(names.begin(), names.end());
		}

		void set_source_location(Node* node, std::source_location loc) {
			if (!node->debug_info) {
				node->debug_info = new NodeDebugInfo;
			}
			node->debug_info->decl_loc = loc;
		}

		// TYPES
		Type* make_imbued_ty(Type* ty, Access access) {
			return emplace_type(Type{ .kind = Type::IMBUED_TY, .size = ty->size, .imbued = { .T = ty, .access = access } });
		}

		Type* make_aliased_ty(Type* ty, size_t ref_idx) {
			return emplace_type(Type{ .kind = Type::ALIASED_TY, .size = ty->size, .aliased = { .T = ty, .ref_idx = ref_idx } });
		}

		Type* u64() {
			return emplace_type(Type{ .kind = Type::INTEGER_TY, .size = sizeof(uint64_t), .integer = { .width = 64 } });
		}

		Type* u32() {
			return emplace_type(Type{ .kind = Type::INTEGER_TY, .size = sizeof(uint32_t), .integer = { .width = 32 } });
		}

		// OPS

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

		template<class T>
		Ref make_constant(T value) {
			Type** ty;
			if constexpr (std::is_same_v<T, uint64_t>) {
				ty = new Type*(u64());
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				ty = new Type*(u32());
			} else {
				ty = new Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			}
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = new T(value) } }));
		}

		Ref make_declare_image(ImageAttachment value) {
			auto ptr = new ImageAttachment(value); /* rest extent_x extent_y extent_z format samples base_layer layer_count base_level level_count */
			auto args_ptr = new Ref[10];
			auto mem_ty = new Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = ptr } }));
			auto u64_ty = new Type*(u64());
			auto u32_ty = new Type*(u32());
			if (value.extent.extent.width > 0) {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u32_ty, 1 }, .constant = { .value = &ptr->extent.extent.width } }));
			} else {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u32_ty, 1 } }));
			}
			if (value.extent.extent.height > 0) {
				args_ptr[2] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u32_ty, 1 }, .constant = { .value = &ptr->extent.extent.height } }));
			} else {
				args_ptr[2] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u32_ty, 1 } }));
			}
			if (value.extent.extent.depth > 0) {
				args_ptr[3] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u32_ty, 1 }, .constant = { .value = &ptr->extent.extent.depth } }));
			} else {
				args_ptr[3] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u32_ty, 1 } }));
			}
			if (value.format != Format::eUndefined) {
				args_ptr[4] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = &ptr->format } }));
			} else {
				args_ptr[4] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ mem_ty, 1 } }));
			}
			if (value.sample_count != Samples::eInfer) {
				args_ptr[5] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = &ptr->sample_count } }));
			} else {
				args_ptr[5] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ mem_ty, 1 } }));
			}
			if (value.base_layer != VK_REMAINING_ARRAY_LAYERS) {
				args_ptr[6] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u32_ty, 1 }, .constant = { .value = &ptr->base_layer } }));
			} else {
				args_ptr[6] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u32_ty, 1 } }));
			}
			if (value.layer_count != VK_REMAINING_ARRAY_LAYERS) {
				args_ptr[7] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u32_ty, 1 }, .constant = { .value = &ptr->layer_count } }));
			} else {
				args_ptr[7] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u32_ty, 1 } }));
			}
			if (value.base_level != VK_REMAINING_MIP_LEVELS) {
				args_ptr[8] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u32_ty, 1 }, .constant = { .value = &ptr->base_level } }));
			} else {
				args_ptr[8] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u32_ty, 1 } }));
			}
			if (value.level_count != VK_REMAINING_MIP_LEVELS) {
				args_ptr[9] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u32_ty, 1 }, .constant = { .value = &ptr->level_count } }));
			} else {
				args_ptr[9] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u32_ty, 1 } }));
			}

			return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ &builtin_image, 1 }, .construct = { .args = std::span(args_ptr, 10) } }));
		}

		Ref make_declare_buffer(Buffer value) {
			auto buf_ptr = new Buffer(value); /* rest size */
			auto args_ptr = new Ref[2];
			auto mem_ty = new Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = buf_ptr } }));
			auto u64_ty = new Type*(u64());
			if (value.size != ~(0u)) {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u64_ty, 1 }, .constant = { .value = &buf_ptr->size } }));
			} else {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u64_ty, 1 } }));
			}

			return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ &builtin_buffer, 1 }, .construct = { .args = std::span(args_ptr, 2) } }));
		}

		Ref make_declare_array(Type* type, std::span<Ref> args, std::span<Ref> defs) {
			auto arr_ty = new Type*(
			    emplace_type(Type{ .kind = Type::ARRAY_TY, .size = args.size() * type->size, .array = { .T = type, .count = args.size(), .stride = type->size } }));
			auto args_ptr = new Ref[args.size() + 1];
			auto mem_ty = new Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = nullptr } }));
			std::copy(args.begin(), args.end(), args_ptr + 1);
			auto defs_ptr = new Ref[defs.size()];
			std::copy(defs.begin(), defs.end(), defs_ptr);
			return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
			                              .type = std::span{ arr_ty, 1 },
			                              .construct = { .args = std::span(args_ptr, args.size() + 1), .defs = std::span(defs_ptr, defs.size()) } }));
		}

		Ref make_declare_swapchain(Swapchain& bundle) {
			auto swp_ptr = new Swapchain(bundle);
			auto args_ptr = new Ref[2];
			auto mem_ty = new Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = swp_ptr } }));
			std::vector<Ref> imgs;
			for (auto i = 0; i < bundle.images.size(); i++) {
				imgs.push_back(make_declare_image(bundle.images[i]));
			}
			args_ptr[1] = make_declare_array(builtin_image, imgs, {});
			return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ &builtin_swapchain, 1 }, .construct = { .args = std::span(args_ptr, 2) } }));
		}

		Ref make_extract(Ref composite, Ref index) {
			auto stripped = Type::stripped(composite.type());
			assert(stripped->kind == Type::ARRAY_TY);
			auto ty = new Type*(stripped->array.T);
			return first(emplace_op(Node{ .kind = Node::EXTRACT, .type = std::span{ ty, 1 }, .extract = { .composite = composite, .index = index } }));
		}

		Ref make_extract(Ref composite, uint64_t index) {
			auto ty = new Type*;
			auto stripped = Type::stripped(composite.type());
			if (stripped->kind == Type::ARRAY_TY) {
				*ty = stripped->array.T;
			} else if (stripped->kind == Type::COMPOSITE_TY) {
				*ty = stripped->composite.types[index];
			}
			return first(emplace_op(
			    Node{ .kind = Node::EXTRACT, .type = std::span{ ty, 1 }, .extract = { .composite = composite, .index = make_constant<uint64_t>(index) } }));
		}

		Ref make_cast(Type* dst_type, Ref src) {
			auto ty = new Type*(dst_type);
			return first(emplace_op(Node{ .kind = Node::CAST, .type = std::span{ ty, 1 }, .cast = { .src = src } }));
		}

		Ref make_acquire_next_image(Ref swapchain) {
			return first(
			    emplace_op(Node{ .kind = Node::ACQUIRE_NEXT_IMAGE, .type = std::span{ &builtin_image, 1 }, .acquire_next_image = { .swapchain = swapchain } }));
		}

		Ref make_clear_image(Ref dst, Clear cv) {
			return first(emplace_op(Node{ .kind = Node::CLEAR, .type = std::span{ &builtin_image, 1 }, .clear = { .dst = dst, .cv = new Clear(cv) } }));
		}

		Type* make_opaque_fn_ty(std::span<Type* const> args,
		                        std::span<Type* const> ret_types,
		                        DomainFlags execute_on,
		                        std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>, std::span<void*>)> callback) {
			auto arg_ptr = new Type*[args.size()];
			std::copy(args.begin(), args.end(), arg_ptr);
			auto ret_ty_ptr = new Type*[ret_types.size()];
			std::copy(ret_types.begin(), ret_types.end(), ret_ty_ptr);
			return emplace_type(
			    Type{ .kind = Type::OPAQUE_FN_TY,
			          .opaque_fn = { .args = std::span(arg_ptr, args.size()),
			                         .return_types = std::span(ret_ty_ptr, ret_types.size()),
			                         .execute_on = execute_on.m_mask,
			                         .callback = new std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>, std::span<void*>)>(callback) } });
		}

		Ref make_declare_fn(Type* const fn_ty) {
			auto ty = new Type*(fn_ty);
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = nullptr } }));
		}

		template<class... Refs>
		Node* make_call(Ref fn, Refs... args) {
			Ref* args_ptr = new Ref[sizeof...(args)]{ args... };
			decltype(Node::call) call = { .args = std::span(args_ptr, sizeof...(args)), .fn = fn };
			Node n{};
			n.kind = Node::CALL;
			n.type = fn.type()->opaque_fn.return_types;
			n.call = call;
			return emplace_op(n);
		}

		Node* make_release(Ref src, AcquireRelease* acq_rel, Access dst_access, DomainFlagBits dst_domain) {
			return emplace_op(Node{ .kind = Node::RELEASE, .release = { .src = src, .release = acq_rel, .dst_access = dst_access, .dst_domain = dst_domain } });
		}

		Ref make_relacq(Ref src, AcquireRelease* acq_rel) {
			auto ty = new Type*(Type::stripped(src.type()));
			return first(emplace_op(Node{ .kind = Node::RELACQ, .type = std::span{ ty, 1 }, .relacq = { .src = src, .rel_acq = acq_rel } }));
		}

		Ref make_acquire(Type* type, AcquireRelease* acq_rel, void* value) {
			auto ty = new Type*(emplace_type(*type));
			auto mem_ty = new Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			return first(emplace_op(
			    Node{ .kind = Node::ACQUIRE,
			          .type = std::span{ ty, 1 },
			          .acquire = { .arg = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = value } })),
			                       .acquire = acq_rel } }));
		}

		// MATH

		Ref make_math_binary_op(Node::BinOp op, Ref a, Ref b) {
			Type** tys = new Type*(a.type());

			return first(emplace_op(Node{ .kind = Node::MATH_BINARY, .type = std::span{ tys, 1 }, .math_binary = { .a = a, .b = b, .op = op } }));
		}
	};

	struct ExtRef {
		ExtRef(std::shared_ptr<RG> module, Ref head) : module(std::move(module)) {
			acqrel = std::make_unique<AcquireRelease>();
			this->head = this->module->make_relacq(head, acqrel.get());
		}

		~ExtRef() {
			if (module && head.node->kind == Node::RELACQ) {
				head.node->relacq.rel_acq = nullptr;
			}
		}

		ExtRef(ExtRef&& o) = default;

		Ref get_head() {
			assert(head.node->kind == Node::RELACQ || head.node->kind == Node::RELEASE);
			return head;
		}

		void to_release(Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eAny) noexcept {
			assert(head.node->kind == Node::RELACQ);
			head.node->kind = Node::RELEASE;
			head.node->release = { .src = head.node->relacq.src, .release = acqrel.get(), .dst_access = access, .dst_domain = domain };
			head.node->type = std::span<Type*>();
		}

		std::shared_ptr<RG> module;
		std::unique_ptr<AcquireRelease> acqrel;

	private:
		Ref head;
	};
} // namespace vuk