#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/RelSpan.hpp"
#include "vuk/Swapchain.hpp"
#include "vuk/SyncPoint.hpp"
#include "vuk/Types.hpp"

#include <deque>
#include <functional>
#include <plf_colony.h>
#include <span>
#include <vector>

namespace vuk {
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
		std::vector<ResourceUse> last_use; // last access performed on resource before signalling
	};

	struct TypeDebugInfo {
		std::string_view name;
	};

	using UserCallbackType = std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>, std::span<void*>)>;

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
				UserCallbackType* callback;
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
				hash_combine_direct(v, IMBUED_TY);
				hash_combine_direct(v, (uint32_t)t->imbued.access);
				return v;
			case ALIASED_TY:
				v = Type::hash(t->aliased.T);
				hash_combine_direct(v, ALIASED_TY);
				hash_combine_direct(v, (uint32_t)t->aliased.ref_idx);
				return v;
			case MEMORY_TY:
				return 0;
			case INTEGER_TY:
				return t->integer.width;
			case ARRAY_TY:
				v = Type::hash(t->array.T);
				hash_combine_direct(v, ARRAY_TY);
				hash_combine_direct(v, (uint32_t)t->array.count);
				return v;
			case COMPOSITE_TY:
				v = COMPOSITE_TY;
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
					return std::string(t->debug_info->name);
				}
				return "composite:" + std::to_string(t->composite.tag);
			case OPAQUE_FN_TY:
				return "ofn";
			}
		}

		~Type() {}
	};

	template<class Type, Access acc, class UniqueT, StringLiteral N>
	size_t Arg<Type, acc, UniqueT, N>::size() const noexcept
	  requires std::is_array_v<Type>
	{
		return def.type()->array.count;
	}

	struct RG;

	struct SchedulingInfo {
		SchedulingInfo(DomainFlags required_domains) : required_domains(required_domains) {}
		SchedulingInfo(DomainFlagBits required_domain) : required_domains(required_domain) {}

		DomainFlags required_domains;
	};

	struct NodeDebugInfo {
		std::span<std::string_view> result_names;
		std::span<std::source_location> trace;
	};

	// struct describing use chains
	struct ChainLink {
		Ref urdef;                 // the first def
		ChainLink* prev = nullptr; // if this came from a previous undef, we link them together
		Ref def;
		RelSpan<Ref> reads;
		Ref undef;
		ChainLink* next = nullptr; // if this links to a def, we link them together
		RelSpan<ChainLink*> child_chains;
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
			SLICE,
			CONVERGE,
			IMPORT,
			CALL,
			CLEAR,
			RESOLVE,
			SIGNAL,
			WAIT,
			ACQUIRE,
			RELEASE,
			RELACQ, // can realise into ACQUIRE, RELEASE or NOP
			ACQUIRE_NEXT_IMAGE,
			CAST,
			MATH_BINARY,
			INDIRECT_DEPEND // utility for dependencies on writes
		} kind;
		std::span<Type*> type;
		NodeDebugInfo* debug_info = nullptr;
		SchedulingInfo* scheduling_info = nullptr;
		ChainLink* links = nullptr;

		uint8_t flag = 0;

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
			struct : Fixed<5> {
				Ref image;
				Ref base_level;
				Ref level_count;
				Ref base_layer;
				Ref layer_count;
			} slice;
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
			struct : Variable {
				std::span<Ref> ref_and_diverged;
				std::span<bool> write;
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
				size_t index;
			} acquire;
			struct : Fixed<1> {
				Ref src;
				AcquireRelease* release;
				Access dst_access;
				DomainFlagBits dst_domain;
				void* value;
			} release;
			struct : Variable {
				std::span<Ref> src;
				AcquireRelease* rel_acq;
				std::span<void*> values;
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
			struct : Fixed<1> {
				Ref rref; // reverse Ref
			} indirect_depend;
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
			case SLICE:
				return "slice";
			case CONVERGE:
				return "converge";
			case INDIRECT_DEPEND:
				return "indir_dep";
			}
			assert(0);
			return "";
		}
	};

	inline Ref first(Node* node) noexcept {
		assert(node->type.size() > 0);
		return { node, 0 };
	}

	inline Ref nth(Node* node, size_t idx) noexcept {
		assert(node->type.size() > idx);
		return { node, idx };
	}

	inline Type* Ref::type() const noexcept {
		return node->type[index];
	}

	inline ChainLink& Ref::link() noexcept {
		return node->links[index];
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
			return eval<T>(ref.node->relacq.src[ref.index]);
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

	template<class T, size_t size>
	struct InlineArena {
		std::unique_ptr<InlineArena> next;
		std::byte arena[size];
		std::byte* base;
		std::byte* cur;

		InlineArena() {
			base = cur = arena;
		}

		void* ensure_space(size_t ns) {
			if ((size - (cur - base)) < ns) {
				grow();
			}
			cur += ns;
			return cur - ns;
		}

		void grow() {
			InlineArena* tail = this;
			while (tail->next != nullptr) {
				tail = tail->next.get();
			}
			tail->next = std::make_unique<InlineArena>();
			base = cur = tail->next->arena;
		}

		T* emplace(T v) {
			return new (ensure_space(sizeof(T))) T(std::move(v));
		}

		std::string_view allocate_string(std::string_view sv) {
			auto dst = ensure_space(sv.size());
			memcpy(dst, sv.data(), sv.size());
			return std::string_view{ (char*)dst, sv.size() };
		}

		template<class U>
		std::span<U> allocate_span(std::span<U> sp) {
			auto dst = std::span((U*)ensure_space(sp.size_bytes()), sp.size());

			std::uninitialized_copy(sp.begin(), sp.end(), dst.begin());
			return dst;
		}

		template<class U>
		std::span<U> allocate_span(std::span<U> sp, size_t sz) {
			auto dst = std::span((U*)ensure_space(sizeof(U) * sz), sz);

			std::uninitialized_copy(sp.begin(), sp.end(), dst.begin());
			return dst;
		}

		template<class U>
		std::span<U> allocate_span(std::vector<U> sp) {
			auto dst = std::span((U*)ensure_space(sp.size() * sizeof(U)), sp.size());

			std::uninitialized_move(sp.begin(), sp.end(), dst.begin());
			return dst;
		}
	};

	struct RG {
		plf::colony<Node> op_arena;
		plf::colony<UserCallbackType> ucbs;

		InlineArena<std::byte, 4 * 1024> payload_arena;
		InlineArena<Type, 16 * sizeof(Type)> type_arena;

		Type* builtin_image = nullptr;
		Type* builtin_buffer = nullptr;
		Type* builtin_swapchain = nullptr;

		Type*& get_builtin_image() {
			if (!builtin_image) {
				auto mem_ty = emplace_type(Type{ .kind = Type::MEMORY_TY });
				auto image_ = new (payload_arena.ensure_space(sizeof(Type* [9]))) Type* [9] {
					u32(), u32(), u32(), mem_ty, mem_ty, u32(), u32(), u32(), u32()
				};
				auto image_offsets = new (payload_arena.ensure_space(sizeof(size_t[9]))) size_t[9]{ offsetof(ImageAttachment, extent) + offsetof(Extent3D, width),
					                                                                                  offsetof(ImageAttachment, extent) + offsetof(Extent3D, height),
					                                                                                  offsetof(ImageAttachment, extent) + offsetof(Extent3D, depth),
					                                                                                  offsetof(ImageAttachment, format),
					                                                                                  offsetof(ImageAttachment, sample_count),
					                                                                                  offsetof(ImageAttachment, base_layer),
					                                                                                  offsetof(ImageAttachment, layer_count),
					                                                                                  offsetof(ImageAttachment, base_level),
					                                                                                  offsetof(ImageAttachment, level_count) };
				builtin_image = emplace_type(Type{ .kind = Type::COMPOSITE_TY,
				                                   .size = sizeof(ImageAttachment),
				                                   .debug_info = allocate_type_debug_info("image"),
				                                   .composite = { .types = { image_, 9 }, .offsets = { image_offsets, 9 }, .tag = 0 } });
			}
			return builtin_image;
		}

		Type*& get_builtin_buffer() {
			if (!builtin_buffer) {
				auto buffer_ = new (payload_arena.ensure_space(sizeof(Type* [1]))) Type* [1] {
					u32()
				};
				auto buffer_offsets = new (payload_arena.ensure_space(sizeof(size_t[1]))) size_t[1]{ offsetof(Buffer, size) };
				builtin_buffer = emplace_type(Type{ .kind = Type::COMPOSITE_TY,
				                                    .size = sizeof(Buffer),
				                                    .debug_info = allocate_type_debug_info("buffer"),
				                                    .composite = { .types = { buffer_, 1 }, .offsets = { buffer_offsets, 1 }, .tag = 1 } });
			}
			return builtin_buffer;
		}

		Type*& get_builtin_swapchain() {
			if (!builtin_swapchain) {
				auto arr_ty = emplace_type(Type{ .kind = Type::ARRAY_TY,
				                                 .size = 16 * get_builtin_image()->size,
				                                 .array = { .T = get_builtin_image(), .count = 16, .stride = get_builtin_image()->size } });
				auto swp_ = new (payload_arena.ensure_space(sizeof(Type* [1]))) Type* [1] {
					arr_ty
				};
				builtin_swapchain = emplace_type(Type{ .kind = Type::COMPOSITE_TY,
				                                       .size = sizeof(Swapchain),
				                                       .debug_info = allocate_type_debug_info("swapchain"),
				                                       .composite = { .types = { swp_, 1 }, .tag = 2 } });
			}
			return builtin_swapchain;
		}

		std::vector<std::shared_ptr<RG>> subgraphs;
		// uint64_t current_hash = 0;

		Node* emplace_op(Node v) {
			return &*op_arena.emplace(std::move(v));
		}

		Type* emplace_type(Type t) {
			return type_arena.emplace(std::move(t));
		}

		void reference_RG(std::shared_ptr<RG> other) {
			subgraphs.emplace_back(std::move(other));
		}

		TypeDebugInfo* allocate_type_debug_info(std::string_view name) {
			return new (payload_arena.ensure_space(sizeof(TypeDebugInfo))) TypeDebugInfo{ payload_arena.allocate_string(name) };
		}

		void name_output(Ref ref, std::string_view name) {
			auto node = ref.node;
			if (!node->debug_info) {
				node->debug_info = new (payload_arena.ensure_space(sizeof(NodeDebugInfo))) NodeDebugInfo;
			}
			auto& names = ref.node->debug_info->result_names;
			if (names.size() <= ref.index) {
				names = payload_arena.allocate_span(names, ref.index + 1);
			}
			names[ref.index] = payload_arena.allocate_string(name);
		}

		void set_source_location(Node* node, SourceLocationAtFrame loc) {
			if (!node->debug_info) {
				node->debug_info = new (payload_arena.ensure_space(sizeof(NodeDebugInfo))) NodeDebugInfo;
			}
			auto p = &loc;
			size_t cnt = 0;
			do {
				cnt++;
				p = p->parent;
			} while (p != nullptr);
			node->debug_info->trace = std::span((vuk::source_location*)payload_arena.ensure_space(sizeof(vuk::source_location) * cnt), cnt);
			p = &loc;
			cnt = 0;
			do {
				node->debug_info->trace[cnt] = p->location;
				cnt++;
				p = p->parent;
			} while (p != nullptr);
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

		template<class T>
		Ref make_constant(T value) {
			Type** ty;
			if constexpr (std::is_same_v<T, uint64_t>) {
				ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(u64());
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(u32());
			} else {
				ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			}
			return first(emplace_op(
			    Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = new (payload_arena.ensure_space(sizeof(T))) T(value) } }));
		}

		Ref make_declare_image(ImageAttachment value) {
			auto ptr = new (payload_arena.ensure_space(sizeof(ImageAttachment)))
			    ImageAttachment(value); /* rest extent_x extent_y extent_z format samples base_layer layer_count base_level level_count */
			auto args_ptr = new (payload_arena.ensure_space(sizeof(Ref) * 10)) Ref[10];
			auto mem_ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = ptr } }));
			auto u32_ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(u32());
			if (value.extent.width > 0) {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u32_ty, 1 }, .constant = { .value = &ptr->extent.width } }));
			} else {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u32_ty, 1 } }));
			}
			if (value.extent.height > 0) {
				args_ptr[2] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u32_ty, 1 }, .constant = { .value = &ptr->extent.height } }));
			} else {
				args_ptr[2] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u32_ty, 1 } }));
			}
			if (value.extent.depth > 0) {
				args_ptr[3] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u32_ty, 1 }, .constant = { .value = &ptr->extent.depth } }));
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

			return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ &get_builtin_image(), 1 }, .construct = { .args = std::span(args_ptr, 10) } }));
		}

		Ref make_declare_buffer(Buffer value) {
			auto buf_ptr = new (payload_arena.ensure_space(sizeof(Buffer))) Buffer(value); /* rest size */
			auto args_ptr = new (payload_arena.ensure_space(sizeof(Ref[2]))) Ref[2];
			auto mem_ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = buf_ptr } }));
			auto u64_ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(u64());
			if (value.size != ~(0u)) {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ u64_ty, 1 }, .constant = { .value = &buf_ptr->size } }));
			} else {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ u64_ty, 1 } }));
			}

			return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ &get_builtin_buffer(), 1 }, .construct = { .args = std::span(args_ptr, 2) } }));
		}

		Ref make_declare_array(Type* type, std::span<Ref> args, std::span<Ref> defs) {
			auto arr_ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(
			    emplace_type(Type{ .kind = Type::ARRAY_TY, .size = args.size() * type->size, .array = { .T = type, .count = args.size(), .stride = type->size } }));
			auto args_ptr = new (payload_arena.ensure_space(sizeof(Ref) * (args.size() + 1))) Ref[args.size() + 1];
			auto mem_ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = nullptr } }));
			std::copy(args.begin(), args.end(), args_ptr + 1);
			auto defs_ptr = new (payload_arena.ensure_space(sizeof(Ref) * defs.size())) Ref[defs.size()];
			std::copy(defs.begin(), defs.end(), defs_ptr);
			return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
			                              .type = std::span{ arr_ty, 1 },
			                              .construct = { .args = std::span(args_ptr, args.size() + 1), .defs = std::span(defs_ptr, defs.size()) } }));
		}

		Ref make_declare_swapchain(Swapchain& bundle) {
			auto args_ptr = new (payload_arena.ensure_space(sizeof(Ref[2]))) Ref[2];
			auto mem_ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = &bundle } }));
			std::vector<Ref> imgs;
			for (auto i = 0; i < bundle.images.size(); i++) {
				imgs.push_back(make_declare_image(bundle.images[i]));
			}
			args_ptr[1] = make_declare_array(get_builtin_image(), imgs, {});
			return first(
			    emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ &get_builtin_swapchain(), 1 }, .construct = { .args = std::span(args_ptr, 2) } }));
		}

		Ref make_extract(Ref composite, Ref index) {
			auto stripped = Type::stripped(composite.type());
			assert(stripped->kind == Type::ARRAY_TY);
			auto ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(stripped->array.T);
			return first(emplace_op(Node{ .kind = Node::EXTRACT, .type = std::span{ ty, 1 }, .extract = { .composite = composite, .index = index } }));
		}

		Ref make_extract(Ref composite, uint64_t index) {
			auto ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*;
			auto stripped = Type::stripped(composite.type());
			if (stripped->kind == Type::ARRAY_TY) {
				*ty = stripped->array.T;
			} else if (stripped->kind == Type::COMPOSITE_TY) {
				*ty = stripped->composite.types[index];
			}
			return first(emplace_op(
			    Node{ .kind = Node::EXTRACT, .type = std::span{ ty, 1 }, .extract = { .composite = composite, .index = make_constant<uint64_t>(index) } }));
		}

		Ref make_slice(Ref image, Ref base_level, Ref level_count, Ref base_layer, Ref layer_count) {
			auto stripped = Type::stripped(image.type());
			auto ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(stripped);
			return first(emplace_op(
			    Node{ .kind = Node::SLICE,
			          .type = std::span{ ty, 1 },
			          .slice = { .image = image, .base_level = base_level, .level_count = level_count, .base_layer = base_layer, .layer_count = layer_count } }));
		}

		Ref make_converge(Ref ref, std::span<Ref> deps, std::span<char> write) {
			auto stripped = Type::stripped(ref.type());
			auto ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(stripped);

			auto deps_ptr = new (payload_arena.ensure_space(sizeof(Ref) * (deps.size() + 1))) Ref[deps.size() + 1];
			deps_ptr[0] = ref;
			std::copy(deps.begin(), deps.end(), deps_ptr + 1);
			auto rw_ptr = new (payload_arena.ensure_space(sizeof(bool) * (deps.size()))) bool[deps.size()];
			std::copy(write.begin(), write.end(), rw_ptr);
			return first(emplace_op(Node{ .kind = Node::CONVERGE,
			                              .type = std::span{ ty, 1 },
			                              .converge = { .ref_and_diverged = std::span{ deps_ptr, deps.size() + 1 }, .write = std::span{ rw_ptr, deps.size() } } }));
		}

		Ref make_cast(Type* dst_type, Ref src) {
			auto ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(dst_type);
			return first(emplace_op(Node{ .kind = Node::CAST, .type = std::span{ ty, 1 }, .cast = { .src = src } }));
		}

		Ref make_acquire_next_image(Ref swapchain) {
			return first(
			    emplace_op(Node{ .kind = Node::ACQUIRE_NEXT_IMAGE, .type = std::span{ &get_builtin_image(), 1 }, .acquire_next_image = { .swapchain = swapchain } }));
		}

		Ref make_clear_image(Ref dst, Clear cv) {
			return first(emplace_op(Node{ .kind = Node::CLEAR, .type = std::span{ &get_builtin_image(), 1 }, .clear = { .dst = dst, .cv = new Clear(cv) } }));
		}

		Type* make_opaque_fn_ty(std::span<Type* const> args, std::span<Type* const> ret_types, DomainFlags execute_on, UserCallbackType callback) {
			auto arg_ptr = new (payload_arena.ensure_space(sizeof(Type*) * args.size())) Type*[args.size()];
			std::copy(args.begin(), args.end(), arg_ptr);
			auto ret_ty_ptr = new (payload_arena.ensure_space(sizeof(Type*) * ret_types.size())) Type*[ret_types.size()];
			std::copy(ret_types.begin(), ret_types.end(), ret_ty_ptr);
			return emplace_type(Type{ .kind = Type::OPAQUE_FN_TY,
			                          .opaque_fn = { .args = std::span(arg_ptr, args.size()),
			                                         .return_types = std::span(ret_ty_ptr, ret_types.size()),
			                                         .execute_on = execute_on.m_mask,
			                                         .callback = &*ucbs.emplace(std::move(callback)) } });
		}

		Ref make_declare_fn(Type* const fn_ty) {
			auto ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(fn_ty);
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = nullptr } }));
		}

		template<class... Refs>
		Node* make_call(Ref fn, Refs... args) {
			Ref* args_ptr = new (payload_arena.ensure_space(sizeof(Ref[sizeof...(args)]))) Ref[sizeof...(args)]{ args... };
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

		Node* make_relacq(Node* src, AcquireRelease* acq_rel) {
			Ref* args_ptr = new (payload_arena.ensure_space(sizeof(Ref) * src->type.size())) Ref[src->type.size()];
			auto tys = new (payload_arena.ensure_space(sizeof(Type*) * src->type.size())) Type*[src->type.size()];
			for (size_t i = 0; i < src->type.size(); i++) {
				args_ptr[i] = Ref{ src, i };
				tys[i] = Type::stripped(src->type[i]);
			}
			return emplace_op(Node{
			    .kind = Node::RELACQ, .type = std::span{ tys, src->type.size() }, .relacq = { .src = std::span{ args_ptr, src->type.size() }, .rel_acq = acq_rel } });
		}

		Type* copy_type(Type* type) {
			auto make_type_copy = [this](Type*& t) {
				t = emplace_type(*t);
			};
			// copy outer type, then copy inner types as needed
			make_type_copy(type);

			if (type->kind == Type::ALIASED_TY) {
				make_type_copy(type->aliased.T);
			} else if (type->kind == Type::IMBUED_TY) {
				make_type_copy(type->imbued.T);
			} else if (type->kind == Type::ARRAY_TY) {
				make_type_copy(type->array.T);
			} else if (type->kind == Type::COMPOSITE_TY) {
				auto type_array = new (payload_arena.ensure_space(sizeof(Type*) * type->composite.types.size())) Type*[type->composite.types.size()];
				for (auto i = 0; i < type->composite.types.size(); i++) {
					type_array[i] = emplace_type(*type->composite.types[i]);
				}
				type->composite.types = { type_array, type->composite.types.size() };
			}
			return type;
		}

		Ref make_acquire(Type* type, AcquireRelease* acq_rel, size_t index, void* value) {
			auto ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(copy_type(type));
			auto mem_ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(emplace_type(Type{ .kind = Type::MEMORY_TY }));
			return first(emplace_op(
			    Node{ .kind = Node::ACQUIRE,
			          .type = std::span{ ty, 1 },
			          .acquire = { .arg = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = value } })),
			                       .acquire = acq_rel,
			                       .index = index } }));
		}

		template<class T>
		Ref make_acquire(Type* type, AcquireRelease* acq_rel, T value) {
			auto val_ptr = new (payload_arena.ensure_space(sizeof(T))) T(value);
			return make_acquire(type, acq_rel, 0, (void*)val_ptr);
		}

		Ref make_indirect_depend(Node* node, size_t index) {
			Ref true_ref;
			Type* type = nullptr;
			auto count = node->generic_node.arg_count;
			if (count != (uint8_t)~0u) {
				true_ref = node->fixed_node.args[index];
			} else {
				if (node->kind == Node::CALL) {
					type = node->call.fn.type()->opaque_fn.args[index];
				}
				true_ref = node->variable_node.args[index];
			}
			if (!type) {
				type = true_ref.type();
			}
			auto ty = new (payload_arena.ensure_space(sizeof(Type*))) Type*(type);
			return first(emplace_op(Node{ .kind = Node::INDIRECT_DEPEND, .type = std::span{ ty, 1 }, .indirect_depend = { .rref = { node, index } } }));
		}

		// MATH

		Ref make_math_binary_op(Node::BinOp op, Ref a, Ref b) {
			Type** tys = new (payload_arena.ensure_space(sizeof(Type*))) Type*(a.type());

			return first(emplace_op(Node{ .kind = Node::MATH_BINARY, .type = std::span{ tys, 1 }, .math_binary = { .a = a, .b = b, .op = op } }));
		}
	};

	struct ExtNode {
		ExtNode(std::shared_ptr<RG> module, Node* node) : module(std::move(module)) {
			owned_acqrel = std::make_unique<AcquireRelease>();
			acqrel = owned_acqrel.get();
			if (node->kind != Node::RELEASE && node->kind != Node::ACQUIRE) {
				this->node = this->module->make_relacq(node, acqrel);
			} else {
				this->node = node;
			}
		}

		~ExtNode() {
			if (module) {
				if (node->kind == Node::RELACQ) {
					node->relacq.rel_acq = nullptr;
					for (auto i = 0; i < node->relacq.values.size(); i++) {
						auto& v = node->relacq.values[i];
						if (node->relacq.src[i].type() == module->builtin_buffer) {
							delete (Buffer*)v;
						} else {
							delete (ImageAttachment*)v;
						}
					}

					delete node->relacq.values.data();
				} else if (node->kind == Node::RELEASE) {
					if (node->release.src.type() == module->builtin_buffer) {
						delete (Buffer*)node->release.value;
					} else {
						delete (ImageAttachment*)node->release.value;
					}
				}
			}
		}

		ExtNode(ExtNode&& o) = default;

		Node* get_node() {
			assert(node->kind == Node::RELACQ || node->kind == Node::RELEASE || node->kind == Node::ACQUIRE);
			return node;
		}

		std::shared_ptr<RG> module;
		AcquireRelease* acqrel;
		std::unique_ptr<AcquireRelease> owned_acqrel;

	private:
		Node* node;
	};

	struct ExtRef {
		ExtRef(std::shared_ptr<ExtNode> node, Ref ref) : node(node), index(ref.index) {}

		std::shared_ptr<ExtNode> node;
		size_t index;
	};
} // namespace vuk