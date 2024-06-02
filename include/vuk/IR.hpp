#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/RelSpan.hpp"
#include "vuk/ResourceUse.hpp"
#include "vuk/ShortAlloc.hpp"
#include "vuk/SourceLocation.hpp"
#include "vuk/SyncPoint.hpp"
#include "vuk/Types.hpp"
#include "vuk/runtime/vk/VkSwapchain.hpp" //TODO: leaking vk

#include <deque>
#include <functional>
#include <optional>
#include <plf_colony.h>
#include <span>
#include <unordered_map>
#include <vector>

namespace vuk {
	struct TypeDebugInfo {
		std::string name;
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

		static std::string_view to_sv(Access acc) {
			switch (acc) {
			case eNone:
				return "None";
			case eClear:
				return "Clear";
			case eColorWrite:
				return "ColorW";
			case eColorRead:
				return "ColorR";
			case eColorRW:
				return "ColorRW";
			case eDepthStencilRead:
				return "DSRead";
			case eDepthStencilWrite:
				return "DSWrite";
			case eDepthStencilRW:
				return "DSRW";
			case eInputRead:
				return "InputR";
			case eVertexSampled:
				return "VtxS";
			case eVertexRead:
				return "VtxR";
			case eAttributeRead:
				return "AttrR";
			case eIndexRead:
				return "IdxR";
			case eIndirectRead:
				return "IndirR";
			case eFragmentSampled:
				return "FragS";
			case eFragmentRead:
				return "FragR";
			case eFragmentWrite:
				return "FragW";
			case eFragmentRW:
				return "FragRW";
			case eTransferRead:
				return "XferR";
			case eTransferWrite:
				return "XferW";
			case eTransferRW:
				return "XferRW";
			case eComputeRead:
				return "CompR";
			case eComputeWrite:
				return "CompW";
			case eComputeRW:
				return "CompRW";
			case eComputeSampled:
				return "CompS";
			case eRayTracingRead:
				return "RTR";
			case eRayTracingWrite:
				return "RTW";
			case eRayTracingRW:
				return "RTRW";
			case eRayTracingSampled:
				return "RTS";
			case eAccelerationStructureBuildRead:
				return "ASBuildR";
			case eAccelerationStructureBuildWrite:
				return "ASBuildW";
			case eAccelerationStructureBuildRW:
				return "ASBuildRW";
			case eHostRead:
				return "HostR";
			case eHostWrite:
				return "HostW";
			case eHostRW:
				return "HostRW";
			case eMemoryRead:
				return "MemR";
			case eMemoryWrite:
				return "MemW";
			case eMemoryRW:
				return "MemRW";
			}
		}

		static std::string to_string(Type* t) {
			switch (t->kind) {
			case IMBUED_TY:
				return to_string(t->imbued.T) + std::string(":") + std::string(to_sv(t->imbued.access));
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

	struct IRModule;

	struct SchedulingInfo {
		SchedulingInfo(DomainFlags required_domains) : required_domains(required_domains) {}
		SchedulingInfo(DomainFlagBits required_domain) : required_domains(required_domain) {}

		DomainFlags required_domains;
	};

	struct NodeDebugInfo {
		std::span<std::string_view> result_names;
		std::span<vuk::source_location> trace;
	};

	// struct describing use chains
	struct ChainLink {
		Ref urdef = {};            // the first def
		ChainLink* prev = nullptr; // if this came from a previous undef, we link them together
		Ref def;
		RelSpan<Ref> reads;
		Ref undef;
		ChainLink* next = nullptr; // if this links to a def, we link them together
		RelSpan<ChainLink*> child_chains;
		std::optional<ResourceUse> read_sync;  // optional, half sync to put resource into read
		std::optional<ResourceUse> undef_sync; // optional, half sync to put resource into write
	};

	struct ExecutionInfo;

	struct Node {
		static constexpr uint8_t MAX_ARGS = 5;

		enum class BinOp { ADD, SUB, MUL, DIV, MOD };
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
			SPLICE, // for joining subgraphs - can morph into ACQUIRE or NOP, depending on the subgraph state
			ACQUIRE_NEXT_IMAGE,
			CAST,
			MATH_BINARY,
			INDIRECT_DEPEND // utility for dependencies on writes
		} kind;
		uint8_t flag = 0;
		std::span<Type*> type;
		NodeDebugInfo* debug_info = nullptr;
		SchedulingInfo* scheduling_info = nullptr;
		ChainLink* links = nullptr;
		ExecutionInfo* execution_info = nullptr;

		template<uint8_t c>
		struct Fixed {
			static_assert(c <= MAX_ARGS);
			uint8_t arg_count = c;
		};

		struct Variable {
			uint8_t arg_count = (uint8_t)~0u;
		};

		union {
			struct : Fixed<0> {
			} placeholder;
			struct : Fixed<0> {
				union {
					void* value;
					uint64_t* value_uint64_t;
					uint32_t* value_uint32_t;
				};
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
			struct : Fixed<0> {
				void* value;
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
			} splice;
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
			case NOP:
				return "nop";
			case PLACEHOLDER:
				return "placeholder";
			case CONSTANT:
				return "constant";
			case IMPORT:
				return "import";
			case CONSTRUCT:
				return "construct";
			case ACQUIRE_NEXT_IMAGE:
				return "acquire_next_image";
			case CALL:
				return "call";
			case EXTRACT:
				return "extract";
			case ACQUIRE:
				return "acquire";
			case SPLICE:
				return "splice";
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
	  requires(std::is_arithmetic_v<T>)
	T eval(Ref ref);

	struct RefOrValue {
		Ref ref;
		void* value;
		bool is_ref;

		static RefOrValue from_ref(Ref r) {
			return { r, nullptr, true };
		}
		static RefOrValue from_value(void* v) {
			return { {}, v, false };
		}
	};

	inline RefOrValue get_def(Ref ref) {
		switch (ref.node->kind) {
		case Node::ACQUIRE:
		case Node::CONSTRUCT:
		case Node::CONSTANT:
			return RefOrValue::from_ref(ref);
		case Node::SPLICE: {
			if (ref.node->splice.rel_acq == nullptr || ref.node->splice.rel_acq->status == Signal::Status::eDisarmed) {
				return get_def(ref.node->splice.src[ref.index]);
			} else {
				return RefOrValue::from_value(ref.node->splice.values[ref.index]);
			}
		}
		case Node::CALL: {
			Type* t = ref.type();
			if (t->kind != Type::ALIASED_TY) {
				throw CannotBeConstantEvaluated(ref);
			}
			return get_def(ref.node->call.args[t->aliased.ref_idx]);
		}
		case Node::EXTRACT: {
			auto index = eval<uint64_t>(ref.node->extract.index);

			auto type = ref.node->extract.composite.type();
			auto composite = get_def(ref.node->extract.composite);
			if (composite.is_ref) {
				if (composite.ref.node->kind == Node::CONSTRUCT) {
					return RefOrValue::from_ref(composite.ref.node->construct.args[index + 1]);
				} else if (composite.ref.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
					auto swp = composite.ref.node->acquire_next_image.swapchain;
					if (swp.node->kind == Node::CONSTRUCT) {
						auto arr = swp.node->construct.args[1]; // array of images
						if (arr.node->kind == Node::CONSTRUCT) {
							auto elem = arr.node->construct.args[1]; // first image
							if (elem.node->kind == Node::CONSTRUCT) {
								return RefOrValue::from_ref(elem.node->construct.args[index + 1]);
							}
						}
					}
				} else {
					throw CannotBeConstantEvaluated{ ref };
				}
			} else {
				if (type->kind == Type::COMPOSITE_TY) {
					auto offset = type->composite.offsets[index];
					return RefOrValue::from_value(reinterpret_cast<void*>(static_cast<unsigned char*>(composite.value) + offset));
				} else if (type->kind == Type::ARRAY_TY) {
					auto offset = type->array.stride * index;
					return RefOrValue::from_value(reinterpret_cast<void*>(static_cast<unsigned char*>(composite.value) + offset));
				} else {
					throw CannotBeConstantEvaluated{ ref };
				}
			}
		}
		case Node::RELEASE:
			if (ref.node->release.release == nullptr || ref.node->release.release->status == Signal::Status::eDisarmed) {
				return get_def(ref.node->release.src);
			} else {
				return RefOrValue::from_value(ref.node->release.value);
			}
		default:
			throw CannotBeConstantEvaluated{ ref };
		}
	}

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
			return static_cast<T>(ref.node->acquire.value);
		}
		case Node::SPLICE: {
			if (ref.node->splice.rel_acq->status == Signal::Status::eDisarmed) {
				return eval<T>(ref.node->splice.src[ref.index]);
			} else {
				return static_cast<T>(ref.node->splice.values[ref.index]);
			}
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
			case Node::BinOp::ADD: {
				return eval<T>(math_binary.a) + eval<T>(math_binary.b);
			}
			case Node::BinOp::SUB: {
				return eval<T>(math_binary.a) - eval<T>(math_binary.b);
			}
			case Node::BinOp::MUL: {
				return eval<T>(math_binary.a) * eval<T>(math_binary.b);
			}
			case Node::BinOp::DIV: {
				return eval<T>(math_binary.a) / eval<T>(math_binary.b);
			}
			case Node::BinOp::MOD: {
				return eval<T>(math_binary.a) % eval<T>(math_binary.b);
			}
			}
			assert(0);
		}

		case Node::EXTRACT: {
			auto def = get_def(ref);
			if (def.is_ref) {
				return eval<T>(def.ref);
			} else {
				return *static_cast<T*>(def.value);
			}
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
			} else if (node->kind == Node::SPLICE) {
			  return get_constant_value(node->splice.src.node);
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

		void reset() {
			next.reset();
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

	template<class T, size_t sz>
	class inline_alloc {
		InlineArena<std::byte, sz>* a_ = nullptr;

	public:
		typedef T value_type;

	public:
		template<class _Up>
		struct rebind {
			typedef inline_alloc<_Up, sz> other;
		};

		inline_alloc() {}
		inline_alloc(InlineArena<std::byte, sz>& a) : a_(&a) {}
		template<class U, size_t szz>
		inline_alloc(const inline_alloc<U, szz>& a) noexcept : a_(a.a_) {}
		inline_alloc(const inline_alloc&) = default;
		inline_alloc& operator=(const inline_alloc&) = delete;

		T* allocate(std::size_t n) {
			return reinterpret_cast<T*>(a_->ensure_space(n * sizeof(T)));
		}

		void deallocate(T* p, std::size_t n) noexcept {}

		template<class T1, class U, size_t s1, size_t s2>
		friend bool operator==(const inline_alloc<T1, s1>& x, const inline_alloc<U, s2>& y) noexcept;

		template<class U, size_t s>
		friend class inline_alloc;
	};

	template<class T, class U, size_t s1, size_t s2>
	inline bool operator==(const inline_alloc<T, s1>& x, const inline_alloc<U, s2>& y) noexcept {
		return &x.a_ == &y.a_;
	}

	template<class T, class U, size_t s1, size_t s2>
	inline bool operator!=(const inline_alloc<T, s1>& x, const inline_alloc<U, s2>& y) noexcept {
		return !(x == y);
	}

	struct IRModule {
		IRModule() : op_arena(/**/) {}

		plf::colony<Node /*, inline_alloc<Node, 4 * 1024>*/> op_arena;
		plf::colony<UserCallbackType> ucbs;
		plf::colony<Type*> type_refs;

		Type* builtin_image = nullptr;
		Type* builtin_buffer = nullptr;
		Type* builtin_swapchain = nullptr;

		std::unordered_map<uint32_t, Type> type_map;

		Type*& get_builtin_image() {
			if (!builtin_image) {
				auto u32_t = u32();
				auto mem_ty = emplace_type(Type{ .kind = Type::MEMORY_TY });
				auto image_ = new Type* [9] {
					u32_t, u32_t, u32_t, mem_ty, mem_ty, u32_t, u32_t, u32_t, u32_t
				};
				auto image_offsets = new size_t[9]{ offsetof(ImageAttachment, extent) + offsetof(Extent3D, width),
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
				auto buffer_ = new Type* [1] {
					u32()
				};
				auto buffer_offsets = new size_t[1]{ offsetof(Buffer, size) };
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
				auto swp_ = new Type* [1] {
					arr_ty
				};
				auto offsets = new size_t[1]{ 0 };

				builtin_swapchain = emplace_type(Type{ .kind = Type::COMPOSITE_TY,
				                                       .size = sizeof(Swapchain),
				                                       .debug_info = allocate_type_debug_info("swapchain"),
				                                       .composite = { .types = { swp_, 1 }, .offsets = { offsets, 1 }, .tag = 2 } });
			}
			return builtin_swapchain;
		}

		// uint64_t current_hash = 0;

		Node* emplace_op(Node v) {
			return &*op_arena.emplace(std::move(v));
		}

		Type* emplace_type(Type tt) {
			Type* t = &tt;
			auto unify_type = [&](Type*& t) {
				auto [v, succ] = type_map.try_emplace(Type::hash(t), *t);
				t = &v->second;
			};

			if (t->kind == Type::ALIASED_TY) {
				assert(t->aliased.T->kind != Type::ALIASED_TY);
				unify_type(t->aliased.T);
			} else if (t->kind == Type::IMBUED_TY) {
				unify_type(t->imbued.T);
			} else if (t->kind == Type::ARRAY_TY) {
				unify_type(t->array.T);
			} else if (t->kind == Type::COMPOSITE_TY) {
				for (auto& elem_ty : t->composite.types) {
					unify_type(elem_ty);
				}
			}
			unify_type(t);

			return t;
		}

		TypeDebugInfo* allocate_type_debug_info(std::string_view name) {
			return new TypeDebugInfo{ std::string(name) };
		}

		void name_output(Ref ref, std::string_view name) {
			auto node = ref.node;
			if (!node->debug_info) {
				node->debug_info = new NodeDebugInfo;
			}
			auto& names = ref.node->debug_info->result_names;
			/* if (names.size() <= ref.index) {
			  names = payload_arena.allocate_span(names, ref.index + 1);
			}
			names[ref.index] = payload_arena.allocate_string(name);*/
		}

		void set_source_location(Node* node, SourceLocationAtFrame loc) {
			if (!node->debug_info) {
				node->debug_info = new NodeDebugInfo;
			}
			auto p = &loc;
			size_t cnt = 0;
			do {
				cnt++;
				p = p->parent;
			} while (p != nullptr);
			node->debug_info->trace = std::span(new vuk::source_location[cnt], cnt);
			p = &loc;
			cnt = 0;
			do {
				node->debug_info->trace[cnt] = p->location;
				cnt++;
				p = p->parent;
			} while (p != nullptr);
		}

		void destroy_node(Node* node) {
			switch (node->kind) {
			case Node::CONSTRUCT:
				if (node->type[0] == builtin_image) {
					delete (ImageAttachment*)node->construct.args[0].node->constant.value;
				} else if (node->type[0] == builtin_buffer) {
					delete (Buffer*)node->construct.args[0].node->constant.value;
				}
			}
			delete[] node->type.data();
			if (node->generic_node.arg_count == (uint8_t)~0u) {
				delete[] node->variable_node.args.data();
			}
			if (node->scheduling_info)
				delete node->scheduling_info;
			if (node->debug_info) {
				if (node->debug_info->trace.size() > 0) {
					delete[] node->debug_info->trace.data();
				}
				if (node->debug_info->result_names.size() > 0) {
					delete[] node->debug_info->result_names.data();
				}

				delete node->debug_info;
			}
			
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
				ty = new Type*[1](u64());
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				ty = new Type*[1](u32());
			} else {
				ty = new Type*[1](emplace_type(Type{ .kind = Type::MEMORY_TY }));
			}
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = new T(value) } }));
		}

		template<class T>
		Ref make_constant(T* value) {
			Type** ty;
			if constexpr (std::is_same_v<T, uint64_t>) {
				ty = new Type*[1](u64());
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				ty = new Type*[1](u32());
			} else {
				ty = new Type*[1](emplace_type(Type{ .kind = Type::MEMORY_TY }));
			}
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = value } }));
		}

		Ref make_declare_image(ImageAttachment value) {
			auto ptr = new ImageAttachment(value); /* rest extent_x extent_y extent_z format samples base_layer layer_count base_level level_count */
			auto args_ptr = new Ref[10];
			auto mem_ty = new Type*[1](emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = ptr } }));
			if (value.extent.width > 0) {
				args_ptr[1] = make_constant(&ptr->extent.width);
			} else {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new Type*[1](u32()), 1 } }));
			}
			if (value.extent.height > 0) {
				args_ptr[2] = make_constant(&ptr->extent.height);
			} else {
				args_ptr[2] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new Type*[1](u32()), 1 } }));
			}
			if (value.extent.depth > 0) {
				args_ptr[3] = make_constant(&ptr->extent.depth);
			} else {
				args_ptr[3] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new Type*[1](u32()), 1 } }));
			}
			if (value.format != Format::eUndefined) {
				args_ptr[4] = make_constant(&ptr->format);
			} else {
				args_ptr[4] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new Type*[1](emplace_type(Type{ .kind = Type::MEMORY_TY })), 1 } }));
			}
			if (value.sample_count != Samples::eInfer) {
				args_ptr[5] = make_constant(&ptr->sample_count);
			} else {
				args_ptr[5] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new Type*[1](emplace_type(Type{ .kind = Type::MEMORY_TY })), 1 } }));
			}
			if (value.base_layer != VK_REMAINING_ARRAY_LAYERS) {
				args_ptr[6] = make_constant(&ptr->base_layer);
			} else {
				args_ptr[6] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new Type*[1](u32()), 1 } }));
			}
			if (value.layer_count != VK_REMAINING_ARRAY_LAYERS) {
				args_ptr[7] = make_constant(&ptr->layer_count);
			} else {
				args_ptr[7] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new Type*[1](u32()), 1 } }));
			}
			if (value.base_level != VK_REMAINING_MIP_LEVELS) {
				args_ptr[8] = make_constant(&ptr->base_level);
			} else {
				args_ptr[8] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new Type*[1](u32()), 1 } }));
			}
			if (value.level_count != VK_REMAINING_MIP_LEVELS) {
				args_ptr[9] = make_constant(&ptr->level_count);
			} else {
				args_ptr[9] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new Type*[1](u32()), 1 } }));
			}

			return first(emplace_op(
			    Node{ .kind = Node::CONSTRUCT, .type = std::span{ new Type*[1](get_builtin_image()), 1 }, .construct = { .args = std::span(args_ptr, 10) } }));
		}

		Ref make_declare_buffer(Buffer value) {
			auto buf_ptr = new Buffer(value); /* rest size */
			auto args_ptr = new Ref[2];
			auto mem_ty = new Type*[1](emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = buf_ptr } }));
			if (value.size != ~(0u)) {
				args_ptr[1] = make_constant(&buf_ptr->size);
			} else {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new Type*[1](u64()), 1 } }));
			}

			return first(emplace_op(
			    Node{ .kind = Node::CONSTRUCT, .type = std::span{ new Type*[1](get_builtin_buffer()), 1 }, .construct = { .args = std::span(args_ptr, 2) } }));
		}

		Ref make_declare_array(Type* type, std::span<Ref> args) {
			auto arr_ty = new Type*[1](
			    emplace_type(Type{ .kind = Type::ARRAY_TY, .size = args.size() * type->size, .array = { .T = type, .count = args.size(), .stride = type->size } }));
			auto args_ptr = new Ref[args.size() + 1];
			auto mem_ty = new Type*[1](emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = nullptr } }));
			std::copy(args.begin(), args.end(), args_ptr + 1);
			return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ arr_ty, 1 }, .construct = { .args = std::span(args_ptr, args.size() + 1) } }));
		}

		Ref make_declare_swapchain(Swapchain& bundle) {
			auto args_ptr = new Ref[2];
			auto mem_ty = new Type*[1](emplace_type(Type{ .kind = Type::MEMORY_TY }));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = &bundle } }));
			std::vector<Ref> imgs;
			for (auto i = 0; i < bundle.images.size(); i++) {
				imgs.push_back(make_declare_image(bundle.images[i]));
			}
			args_ptr[1] = make_declare_array(get_builtin_image(), imgs);
			return first(
			    emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ new Type* [1](get_builtin_swapchain()), 1 }, .construct = { .args = std::span(args_ptr, 2) } }));
		}

		Ref make_extract(Ref composite, Ref index) {
			auto stripped = Type::stripped(composite.type());
			assert(stripped->kind == Type::ARRAY_TY);
			auto ty = new Type*[1](stripped->array.T);
			return first(emplace_op(Node{ .kind = Node::EXTRACT, .type = std::span{ ty, 1 }, .extract = { .composite = composite, .index = index } }));
		}

		Ref make_extract(Ref composite, uint64_t index) {
			auto ty = new Type*[1];
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
			auto ty = new Type*[1](stripped);
			return first(emplace_op(
			    Node{ .kind = Node::SLICE,
			          .type = std::span{ ty, 1 },
			          .slice = { .image = image, .base_level = base_level, .level_count = level_count, .base_layer = base_layer, .layer_count = layer_count } }));
		}

		Ref make_converge(Ref ref, std::span<Ref> deps, std::span<char> write) {
			auto stripped = Type::stripped(ref.type());
			auto ty = new Type*[1](stripped);

			auto deps_ptr = new Ref[deps.size() + 1];
			deps_ptr[0] = ref;
			std::copy(deps.begin(), deps.end(), deps_ptr + 1);
			auto rw_ptr = new bool[deps.size()];
			std::copy(write.begin(), write.end(), rw_ptr);
			return first(emplace_op(Node{ .kind = Node::CONVERGE,
			                              .type = std::span{ ty, 1 },
			                              .converge = { .ref_and_diverged = std::span{ deps_ptr, deps.size() + 1 }, .write = std::span{ rw_ptr, deps.size() } } }));
		}

		Ref make_cast(Type* dst_type, Ref src) {
			auto ty = new Type*[1](dst_type);
			return first(emplace_op(Node{ .kind = Node::CAST, .type = std::span{ ty, 1 }, .cast = { .src = src } }));
		}

		Ref make_acquire_next_image(Ref swapchain) {
			return first(emplace_op(Node{
			    .kind = Node::ACQUIRE_NEXT_IMAGE, .type = std::span{ new Type*[1](get_builtin_image()), 1 }, .acquire_next_image = { .swapchain = swapchain } }));
		}

		Ref make_clear_image(Ref dst, Clear cv) {
			return first(
			    emplace_op(Node{ .kind = Node::CLEAR, .type = std::span{ new Type*[1](get_builtin_image()), 1 }, .clear = { .dst = dst, .cv = new Clear(cv) } }));
		}

		Type*
		make_opaque_fn_ty(std::span<Type* const> args, std::span<Type* const> ret_types, DomainFlags execute_on, UserCallbackType callback, std::string_view name) {
			auto arg_ptr = new Type*[args.size()];
			std::copy(args.begin(), args.end(), arg_ptr);
			auto ret_ty_ptr = new Type*[ret_types.size()];
			std::copy(ret_types.begin(), ret_types.end(), ret_ty_ptr);
			auto ty = emplace_type(Type{ .kind = Type::OPAQUE_FN_TY,
			                             .opaque_fn = { .args = std::span(arg_ptr, args.size()),
			                                            .return_types = std::span(ret_ty_ptr, ret_types.size()),
			                                            .execute_on = execute_on.m_mask,
			                                            .callback = &*ucbs.emplace(std::move(callback)) } });
			if (!ty->debug_info) {
				ty->debug_info = allocate_type_debug_info(name);
			}
			return ty;
		}

		Ref make_declare_fn(Type* const fn_ty) {
			auto ty = new Type*[1](fn_ty);
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = nullptr } }));
		}

		template<class... Refs>
		Node* make_call(Ref fn, Refs... args) {
			Ref* args_ptr = new Ref[sizeof...(args) + 1]{ fn, args... };
			decltype(Node::call) call = { .args = std::span(args_ptr, sizeof...(args) + 1) };
			Node n{};
			n.kind = Node::CALL;
			n.type = fn.type()->opaque_fn.return_types;
			n.call = call;
			return emplace_op(n);
		}

		Node* make_release(Ref src, AcquireRelease* acq_rel, Access dst_access, DomainFlagBits dst_domain) {
			auto ty = new Type*[1](Type::stripped(src.type()));
			return emplace_op(Node{ .kind = Node::RELEASE,
			                        .type = std::span{ ty, 1 },
			                        .release = { .src = src, .release = acq_rel, .dst_access = dst_access, .dst_domain = dst_domain } });
		}

		Node* make_splice(Node* src, AcquireRelease* acq_rel) {
			Ref* args_ptr = new Ref[src->type.size()];
			auto tys = new Type*[src->type.size()];
			for (size_t i = 0; i < src->type.size(); i++) {
				args_ptr[i] = Ref{ src, i };
				tys[i] = Type::stripped(src->type[i]);
			}
			return emplace_op(Node{
			    .kind = Node::SPLICE, .type = std::span{ tys, src->type.size() }, .splice = { .src = std::span{ args_ptr, src->type.size() }, .rel_acq = acq_rel } });
		}

		Node* copy_node(Node* node) {
			return emplace_op(*node);
		}

		Ref make_acquire(Type* type, AcquireRelease* acq_rel, size_t index, void* value) {
			auto ty = new Type*[1](type);
			return first(emplace_op(Node{ .kind = Node::ACQUIRE, .type = std::span{ ty, 1 }, .acquire = { .value = value, .acquire = acq_rel, .index = index } }));
		}

		template<class T>
		Ref make_acquire(Type* type, AcquireRelease* acq_rel, T value) {
			auto val_ptr = new T(value);
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
					type = node->call.args[0].type()->opaque_fn.args[index - 1];
				}
				true_ref = node->variable_node.args[index];
			}
			if (!type) {
				type = true_ref.type();
			}
			auto ty = new Type*[1](type);
			return first(emplace_op(Node{ .kind = Node::INDIRECT_DEPEND, .type = std::span{ ty, 1 }, .indirect_depend = { .rref = { node, index } } }));
		}

		// MATH

		Ref make_math_binary_op(Node::BinOp op, Ref a, Ref b) {
			Type** tys = new Type*[1](a.type());

			return first(emplace_op(Node{ .kind = Node::MATH_BINARY, .type = std::span{ tys, 1 }, .math_binary = { .a = a, .b = b, .op = op } }));
		}
	};

	inline thread_local IRModule current_module;

	struct ExtNode {
		ExtNode(Node* node, std::vector<std::shared_ptr<ExtNode>> deps) : deps(std::move(deps)) {
			owned_acqrel = std::make_unique<AcquireRelease>();
			acqrel = owned_acqrel.get();
			if (node->kind != Node::RELEASE && node->kind != Node::ACQUIRE) {
				this->node = current_module.make_splice(node, acqrel);
			} else {
				this->node = node;
			}
		}

		ExtNode(Node* node, std::shared_ptr<ExtNode> dep) {
			owned_acqrel = std::make_unique<AcquireRelease>();
			acqrel = owned_acqrel.get();
			if (node->kind != Node::RELEASE && node->kind != Node::ACQUIRE) {
				this->node = current_module.make_splice(node, acqrel);
			} else {
				this->node = node;
			}

			deps.push_back(std::move(dep));
		}

		~ExtNode() {
			if (owned_acqrel) {
				if (node->kind == Node::SPLICE) {
					node->splice.rel_acq = nullptr;
					for (auto i = 0; i < node->splice.values.size(); i++) {
						auto& v = node->splice.values[i];
						if (node->type[i] == current_module.builtin_buffer) {
							delete (Buffer*)v;
						} else {
							delete (ImageAttachment*)v;
						}
					}

					delete node->splice.values.data();

					if (owned_acqrel->status != Signal::Status::eDisarmed) {
						auto it = current_module.op_arena.get_iterator(node);
						current_module.destroy_node(node);
						current_module.op_arena.erase(it);
					}
				} else if (node->kind == Node::RELEASE) {
					if (node->type[0] == current_module.builtin_buffer) {
						delete (Buffer*)node->release.value;
					} else {
						delete (ImageAttachment*)node->release.value;
					}
					if (owned_acqrel->status != Signal::Status::eDisarmed) {
						auto it = current_module.op_arena.get_iterator(node);
						current_module.destroy_node(node);
						current_module.op_arena.erase(it);
					}
				}
			}
		}

		ExtNode(ExtNode&& o) = default;

		Node* get_node() {
			assert(node->kind == Node::NOP || node->kind == Node::SPLICE || node->kind == Node::RELEASE || node->kind == Node::ACQUIRE);
			return node;
		}

		void mutate(Node* new_node) {
			if (node->kind == Node::SPLICE) {
				node->splice.rel_acq = nullptr;
			}
			node = current_module.make_splice(new_node, acqrel);
		}

		AcquireRelease* acqrel;
		std::unique_ptr<AcquireRelease> owned_acqrel;
		std::vector<std::shared_ptr<ExtNode>> deps;

	private:
		Node* node;
	};

	struct ExtRef {
		ExtRef(std::shared_ptr<ExtNode> node, Ref ref) : node(node), index(ref.index) {}

		std::shared_ptr<ExtNode> node;
		size_t index;
	};
} // namespace vuk
