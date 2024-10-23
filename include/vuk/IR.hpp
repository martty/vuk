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
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

// #define VUK_GARBAGE_SAN

namespace vuk {
	struct TypeDebugInfo {
		std::string name;
	};

	using UserCallbackType = std::function<void(CommandBuffer&, std::span<void*>, std::span<void*>, std::span<void*>)>;

	struct Type {
		enum TypeKind { MEMORY_TY = 1, INTEGER_TY, COMPOSITE_TY, ARRAY_TY, IMBUED_TY, ALIASED_TY, OPAQUE_FN_TY } kind;
		size_t size = ~0ULL;

		TypeDebugInfo debug_info;

		union {
			struct {
				uint32_t width;
			} integer;
			struct {
				std::shared_ptr<Type> T;
				Access access;
			} imbued;
			struct {
				std::shared_ptr<Type> T;
				size_t ref_idx;
			} aliased;
			struct {
				std::vector<std::shared_ptr<Type>> args;
				std::vector<std::shared_ptr<Type>> return_types;
				int execute_on;
				UserCallbackType callback;
			} opaque_fn;
			struct {
				std::shared_ptr<Type> T;
				size_t count;
				size_t stride;
			} array;
			struct {
				std::vector<std::shared_ptr<Type>> types;
				std::vector<size_t> offsets;
				size_t tag;
			} composite;
		};

		~Type() {
			switch (kind) {
			case IMBUED_TY:
				std::destroy_at(&imbued);
				break;
			case ALIASED_TY:
				std::destroy_at(&aliased);
				break;
			case OPAQUE_FN_TY:
				std::destroy_at(&opaque_fn);
				break;
			case ARRAY_TY:
				std::destroy_at(&array);
				break;
			case COMPOSITE_TY:
				std::destroy_at(&composite);
				break;
			}
		}

		static std::shared_ptr<Type> stripped(std::shared_ptr<Type> t) {
			switch (t->kind) {
			case IMBUED_TY:
				return stripped(t->imbued.T);
			case ALIASED_TY:
				return stripped(t->aliased.T);
			default:
				return t;
			}
		}

		static std::shared_ptr<Type> extract(std::shared_ptr<Type> t, size_t index) {
			assert(t->kind == COMPOSITE_TY);
			assert(index < t->composite.types.size());
			return t->composite.types[index];
		}

		using Hash = uint32_t;
		Hash hash_value;

		static Hash hash(Type* t) {
			Hash v = 0;
			assert(t->kind < 100);
			switch (t->kind) {
			case IMBUED_TY:
				v = Type::hash(t->imbued.T.get());
				hash_combine_direct(v, IMBUED_TY);
				hash_combine_direct(v, (uint32_t)t->imbued.access);
				return v;
			case ALIASED_TY:
				v = Type::hash(t->aliased.T.get());
				hash_combine_direct(v, ALIASED_TY);
				hash_combine_direct(v, (uint32_t)t->aliased.ref_idx);
				return v;
			case MEMORY_TY:
				v = 0;
				hash_combine_direct(v, MEMORY_TY);
				hash_combine_direct(v, t->size);
				return v;
			case INTEGER_TY:
				return t->integer.width;
			case ARRAY_TY:
				v = Type::hash(t->array.T.get());
				hash_combine_direct(v, ARRAY_TY);
				hash_combine_direct(v, (uint32_t)t->array.count);
				return v;
			case COMPOSITE_TY: {
				v = COMPOSITE_TY;
				for (int i = 0; i < t->composite.types.size(); i++) {
					hash_combine_direct(v, Type::hash(t->composite.types[i].get()));
				}
				hash_combine_direct(v, (uint32_t)t->composite.tag);
				return v;
			}
			case OPAQUE_FN_TY:
				hash_combine_direct(v, (uintptr_t)t->opaque_fn.callback.target<void(CommandBuffer&, std::span<void*>, std::span<void*>, std::span<void*>)>() >> 32);
				hash_combine_direct(v,
				                    (uintptr_t)t->opaque_fn.callback.target<void(CommandBuffer&, std::span<void*>, std::span<void*>, std::span<void*>)>() & 0xffffffff);
				return v;
			}
		}

		// TODO: handle multiple flags
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
				return to_string(t->imbued.T.get()) + std::string(":") + std::string(to_sv(t->imbued.access));
			case ALIASED_TY:
				return to_string(t->aliased.T.get()) + std::string("@") + std::to_string(t->aliased.ref_idx);
			case MEMORY_TY:
				return "mem";
			case INTEGER_TY:
				return t->integer.width == 32 ? "i32" : "i64";
			case ARRAY_TY:
				return to_string(t->array.T.get()) + "[" + std::to_string(t->array.count) + "]";
			case COMPOSITE_TY:
				if (!t->debug_info.name.empty()) {
					return std::string(t->debug_info.name);
				}
				return "composite:" + std::to_string(t->composite.tag);
			case OPAQUE_FN_TY:
				return "ofn";
			}
		}
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
			SPLICE, // for joining subgraphs
			ACQUIRE_NEXT_IMAGE,
			CAST,
			MATH_BINARY,
			GARBAGE
		} kind;
		uint8_t flag = 0;
		std::span<std::shared_ptr<Type>> type;
		NodeDebugInfo* debug_info = nullptr;
		SchedulingInfo* scheduling_info = nullptr;
		ChainLink* links = nullptr;
		ExecutionInfo* execution_info = nullptr;
		struct ScheduledItem* scheduled_item = nullptr;
		size_t index;

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
				bool owned = false;
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
				std::span<Ref> diverged;
				std::span<bool> write;
				Subrange::Image range;
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
			struct : Variable {
				std::span<Ref> src;
				AcquireRelease* rel_acq;
				std::span<void*> values;
				Access dst_access;
				DomainFlagBits dst_domain;
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
			case SPLICE:
				return "splice";
			case MATH_BINARY:
				return "math_b";
			case SLICE:
				return "slice";
			case CONVERGE:
				return "converge";
			case GARBAGE:
				return "garbage";
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

	inline std::shared_ptr<Type> Ref::type() const noexcept {
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

	inline Result<RefOrValue, CannotBeConstantEvaluated> get_def(Ref ref) {
		switch (ref.node->kind) {
		case Node::CONSTRUCT:
		case Node::CONSTANT:
		case Node::ACQUIRE_NEXT_IMAGE:
		case Node::SLICE:
			return { expected_value, RefOrValue::from_ref(ref) };
		case Node::SPLICE: {
			if (ref.node->splice.rel_acq == nullptr || ref.node->splice.rel_acq->status == Signal::Status::eDisarmed) {
				return get_def(ref.node->splice.src[ref.index]);
			} else {
				return { expected_value, RefOrValue::from_value(ref.node->splice.values[ref.index]) };
			}
		}
		case Node::CALL: {
			auto t = ref.type();
			if (t->kind != Type::ALIASED_TY) {
				return { expected_error, CannotBeConstantEvaluated{ ref } };
			}
			return get_def(ref.node->call.args[t->aliased.ref_idx]);
		}
		case Node::EXTRACT: {
			auto composite = get_def(ref.node->extract.composite);
			return composite;
		}
		default:
			return { expected_error, CannotBeConstantEvaluated{ ref } };
		}
	}

	template<class T>
	  requires(!std::is_pointer_v<T>)
	Result<T, CannotBeConstantEvaluated> eval(Ref ref) {
		switch (ref.node->kind) {
		case Node::CONSTANT: {
			return { expected_value, constant<T>(ref) };
		}
		case Node::MATH_BINARY: {
			if constexpr (std::is_arithmetic_v<T>) {
				auto& math_binary = ref.node->math_binary;
				switch (math_binary.op) {
#define UNWRAP_A(val)                                                                                                                                          \
	auto a_ = eval<T>(val);                                                                                                                                      \
	if (!a_) {                                                                                                                                                   \
		return a_;                                                                                                                                                 \
	}                                                                                                                                                            \
	auto a = *a_;
#define UNWRAP_B(val)                                                                                                                                          \
	auto b_ = eval<T>(val);                                                                                                                                      \
	if (!b_) {                                                                                                                                                   \
		return b_;                                                                                                                                                 \
	}                                                                                                                                                            \
	auto b = *b_;
				case Node::BinOp::ADD: {
					UNWRAP_A(math_binary.a)
					UNWRAP_B(math_binary.b)
					return { expected_value, a + b };
				}
				case Node::BinOp::SUB: {
					UNWRAP_A(math_binary.a)
					UNWRAP_B(math_binary.b)
					return { expected_value, a - b };
				}
				case Node::BinOp::MUL: {
					UNWRAP_A(math_binary.a)
					UNWRAP_B(math_binary.b)
					return { expected_value, a * b };
				}
				case Node::BinOp::DIV: {
					UNWRAP_A(math_binary.a)
					UNWRAP_B(math_binary.b)
					return { expected_value, a / b };
				}
				case Node::BinOp::MOD: {
					UNWRAP_A(math_binary.a)
					UNWRAP_B(math_binary.b)
					return { expected_value, a % b };
				}
				}
			}
			assert(0);
		}

		case Node::EXTRACT: {
			auto composite_ = get_def(ref);
			if (!composite_) {
				return composite_;
			}
			auto composite = *composite_;
			auto index_ = eval<uint64_t>(ref.node->extract.index);
			if (!index_) {
				return index_;
			}
			auto index = *index_;
			auto type = ref.node->extract.composite.type();
			if (composite.is_ref) {
				if (composite.ref.node->kind == Node::CONSTRUCT) {
					return eval<T>(composite.ref.node->construct.args[index + 1]);
				} else if (composite.ref.node->kind == Node::ACQUIRE_NEXT_IMAGE) {
					auto swp_ = get_def(composite.ref.node->acquire_next_image.swapchain);
					if (!swp_) {
						return swp_;
					}
					auto swp = *swp_;
					if (swp.is_ref && swp.ref.node->kind == Node::CONSTRUCT) {
						auto arr = swp.ref.node->construct.args[1]; // array of images
						if (arr.node->kind == Node::CONSTRUCT) {
							auto elem = arr.node->construct.args[1]; // first image
							if (elem.node->kind == Node::CONSTRUCT) {
								return eval<T>(elem.node->construct.args[index + 1]);
							}
						}
					} else {
						return { expected_error, CannotBeConstantEvaluated{ ref } };
					}
				} else if (composite.ref.node->kind == Node::SLICE) {
					auto slice_def_ = get_def(composite.ref.node->slice.image);
					if (!slice_def_) {
						return slice_def_;
					}
					auto slice_def = *slice_def_;
					if (!slice_def.is_ref || slice_def.ref.node->kind != Node::CONSTRUCT) {
						return { expected_error, CannotBeConstantEvaluated{ ref } }; // TODO: this too limited
					}
					if (index < 6) {
						return eval<T>(slice_def.ref.node->construct.args[index + 1]);
					} else {
						assert(false && "NYI");
					}
				} else {
					return { expected_error, CannotBeConstantEvaluated{ ref } };
				}
			} else {
				if (type->kind == Type::COMPOSITE_TY) {
					auto offset = type->composite.offsets[index];
					return { expected_value, *static_cast<T*>(reinterpret_cast<void*>(static_cast<unsigned char*>(composite.value) + offset)) };
				} else if (type->kind == Type::ARRAY_TY) {
					auto offset = type->array.stride * index;
					return { expected_value, *static_cast<T*>(reinterpret_cast<void*>(static_cast<unsigned char*>(composite.value) + offset)) };
				} else {
					return { expected_error, CannotBeConstantEvaluated{ ref } };
				}
			}
		}
		default:
			return { expected_error, CannotBeConstantEvaluated{ ref } };
		}
	}

	template<class T>
	  requires(std::is_pointer_v<T>)
	Result<T, CannotBeConstantEvaluated> eval(Ref ref) {
		switch (ref.node->kind) {
		case Node::CONSTANT: {
			return { expected_value, static_cast<T>(ref.node->constant.value) };
		}
		case Node::CONSTRUCT: {
			return eval<T>(ref.node->construct.args[0]);
		}
		case Node::SPLICE: {
			if (ref.node->splice.rel_acq->status == Signal::Status::eDisarmed) {
				return eval<T>(ref.node->splice.src[ref.index]);
			} else {
				return { expected_value, static_cast<T>(ref.node->splice.values[ref.index]) };
			}
		}
		case Node::ACQUIRE_NEXT_IMAGE: {
			Swapchain* swp = *eval<Swapchain*>(ref.node->acquire_next_image.swapchain);
			return { expected_value, reinterpret_cast<T>(&swp->images[0]) };
		}
		default:
			return { expected_error, CannotBeConstantEvaluated{ ref } };
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

	struct Types {
		std::unordered_map<Type::Hash, std::weak_ptr<Type>> type_map;
		plf::colony<UserCallbackType> ucbs;
		std::shared_mutex lock;

		Type::Hash builtin_image = -1;
		Type::Hash builtin_buffer = -1;
		Type::Hash builtin_swapchain = -1;

		// TYPES
		static std::shared_ptr<Type> make_imbued_ty(std::shared_ptr<Type> ty, Access access) {
			return Types::global().emplace_type(
			    std::shared_ptr<Type>(new Type{ .kind = Type::IMBUED_TY, .size = ty->size, .imbued = { .T = ty, .access = access } }));
		}

		static std::shared_ptr<Type> make_aliased_ty(std::shared_ptr<Type> ty, size_t ref_idx) {
			return Types::global().emplace_type(
			    std::shared_ptr<Type>(new Type{ .kind = Type::ALIASED_TY, .size = ty->size, .aliased = { .T = ty, .ref_idx = ref_idx } }));
		}

		static std::shared_ptr<Type> u64() {
			return Types::global().emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::INTEGER_TY, .size = sizeof(uint64_t), .integer = { .width = 64 } }));
		}

		static std::shared_ptr<Type> u32() {
			return Types::global().emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::INTEGER_TY, .size = sizeof(uint32_t), .integer = { .width = 32 } }));
		}

		static std::shared_ptr<Type> memory(size_t size) {
			return Types::global().emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::MEMORY_TY, .size = size }));
		}

		static std::shared_ptr<Type> get_builtin_image() {
			auto u32_t = u32();
			auto image_ = std::vector<std::shared_ptr<Type>>{ u32_t, u32_t, u32_t, memory(sizeof(Format)), memory(sizeof(Samples)), u32_t, u32_t, u32_t, u32_t };
			auto image_offsets = std::vector<size_t>{ offsetof(ImageAttachment, extent) + offsetof(Extent3D, width),
				                                        offsetof(ImageAttachment, extent) + offsetof(Extent3D, height),
				                                        offsetof(ImageAttachment, extent) + offsetof(Extent3D, depth),
				                                        offsetof(ImageAttachment, format),
				                                        offsetof(ImageAttachment, sample_count),
				                                        offsetof(ImageAttachment, base_layer),
				                                        offsetof(ImageAttachment, layer_count),
				                                        offsetof(ImageAttachment, base_level),
				                                        offsetof(ImageAttachment, level_count) };
			auto image_type = Types::global().emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
			                                                                               .size = sizeof(ImageAttachment),
			                                                                               .debug_info = Types::global().allocate_type_debug_info("image"),
			                                                                               .composite = { .types = image_, .offsets = image_offsets, .tag = 0 } }));

			Types::global().builtin_image = Type::hash(image_type.get());

			return image_type;
		}

		static std::shared_ptr<Type> get_builtin_buffer() {
			auto buffer_ = std::vector<std::shared_ptr<Type>>{ u64() };
			auto buffer_offsets = std::vector<size_t>{ offsetof(Buffer, size) };
			auto buffer_type =
			    Types::global().emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
			                                                                 .size = sizeof(Buffer),
			                                                                 .debug_info = Types::global().allocate_type_debug_info("buffer"),
			                                                                 .composite = { .types = buffer_, .offsets = buffer_offsets, .tag = 1 } }));
			Types::global().builtin_buffer = Type::hash(buffer_type.get());
			return buffer_type;
		}

		static std::shared_ptr<Type> get_builtin_swapchain() {
			auto arr_ty = Types::global().emplace_type(
			    std::shared_ptr<Type>(new Type{ .kind = Type::ARRAY_TY,
			                                    .size = 16 * get_builtin_image()->size,
			                                    .array = { .T = get_builtin_image(), .count = 16, .stride = get_builtin_image()->size } }));
			auto swp_ = std::vector<std::shared_ptr<Type>>{ arr_ty };
			auto offsets = std::vector<size_t>{ 0 };

			auto swapchain_type = Types::global().emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
			                                                                                   .size = sizeof(Swapchain),
			                                                                                   .debug_info = Types::global().allocate_type_debug_info("swapchain"),
			                                                                                   .composite = { .types = swp_, .offsets = offsets, .tag = 2 } }));
			Types::global().builtin_swapchain = Type::hash(swapchain_type.get());
			return swapchain_type;
		}

		std::shared_ptr<Type> emplace_type(std::shared_ptr<Type> t) {
			auto unify_type = [&](std::shared_ptr<Type>& t) {
				auto th = Type::hash(t.get());
				auto [v, succ] = type_map.try_emplace(th, t);
				if (succ) {
					t->hash_value = th;
				} else if (!v->second.lock()) {
					type_map[th] = t;
				}
				t->hash_value = th;
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

		TypeDebugInfo allocate_type_debug_info(std::string name) {
			return TypeDebugInfo(name);
		}

		void collect() {
			for (auto it = type_map.begin(); it != type_map.end();) {
				if (it->second.expired()) {
					it = type_map.erase(it);
				} else {
					++it;
				}
			}
		}

		static Types& global() {
			static Types t;
			return t;
		}
	};

	struct IRModule {
		IRModule() : op_arena(/**/) {}

		plf::colony<Node /*, inline_alloc<Node, 4 * 1024>*/> op_arena;
		std::vector<Node*> garbage;
		std::unordered_map<Node*, size_t> potential_garbage;
		size_t node_counter = 0;

		// uint64_t current_hash = 0;

		Node* emplace_op(Node v) {
			v.index = node_counter++;
			return &*op_arena.emplace(std::move(v));
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
			case Node::CONSTANT: {
				if (node->constant.owned) {
					delete[] (char*)node->constant.value;
				}
				break;
			}
			case Node::CONVERGE: {
				delete[] node->converge.write.data();
				break;
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

			auto it = op_arena.get_iterator(node);
			if (it != op_arena.end()) {
#ifdef VUK_GARBAGE_SAN
				node->kind = Node::GARBAGE;
				node->generic_node.arg_count = 0;
				node->type = {};
#else
				op_arena.erase(it);
#endif
			} else {
				node->kind = Node::GARBAGE;
				node->generic_node.arg_count = 0;
				node->type = {};
			}
		}

		// OPS

		template<class T>
		Ref make_constant(T value) {
			std::shared_ptr<Type>* ty;
			if constexpr (std::is_same_v<T, uint64_t>) {
				ty = new std::shared_ptr<Type>[1](Types::u64());
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				ty = new std::shared_ptr<Type>[1](Types::u32());
			} else {
				ty = new std::shared_ptr<Type>[1](Types::memory(sizeof(T)));
			}
			return first(
			    emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = new (new char[sizeof(T)]) T(value), .owned = true } }));
		}

		template<class T>
		Ref make_constant(T* value) {
			std::shared_ptr<Type>* ty;
			if constexpr (std::is_same_v<T, uint64_t>) {
				ty = new std::shared_ptr<Type>[1](Types::u64());
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				ty = new std::shared_ptr<Type>[1](Types::u32());
			} else {
				ty = new std::shared_ptr<Type>[1](Types::memory(sizeof(T)));
			}
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = value, .owned = false } }));
		}

		Ref make_declare_image(ImageAttachment value) {
			auto ptr = new (new char[sizeof(ImageAttachment)])
			    ImageAttachment(value); /* rest extent_x extent_y extent_z format samples base_layer layer_count base_level level_count */
			auto args_ptr = new Ref[10];
			auto mem_ty = new std::shared_ptr<Type>[1](Types::memory(sizeof(ImageAttachment)));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = ptr, .owned = true } }));
			if (value.extent.width > 0) {
				args_ptr[1] = make_constant(&ptr->extent.width);
			} else {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1](Types::u32()), 1 } }));
			}
			if (value.extent.height > 0) {
				args_ptr[2] = make_constant(&ptr->extent.height);
			} else {
				args_ptr[2] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1](Types::u32()), 1 } }));
			}
			if (value.extent.depth > 0) {
				args_ptr[3] = make_constant(&ptr->extent.depth);
			} else {
				args_ptr[3] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1](Types::u32()), 1 } }));
			}
			if (value.format != Format::eUndefined) {
				args_ptr[4] = make_constant(&ptr->format);
			} else {
				args_ptr[4] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1](Types::memory(sizeof(Format))), 1 } }));
			}
			if (value.sample_count != Samples::eInfer) {
				args_ptr[5] = make_constant(&ptr->sample_count);
			} else {
				args_ptr[5] =
				    first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1](Types::memory(sizeof(Samples))), 1 } }));
			}
			if (value.base_layer != VK_REMAINING_ARRAY_LAYERS) {
				args_ptr[6] = make_constant(&ptr->base_layer);
			} else {
				args_ptr[6] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1](Types::u32()), 1 } }));
			}
			if (value.layer_count != VK_REMAINING_ARRAY_LAYERS) {
				args_ptr[7] = make_constant(&ptr->layer_count);
			} else {
				args_ptr[7] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1](Types::u32()), 1 } }));
			}
			if (value.base_level != VK_REMAINING_MIP_LEVELS) {
				args_ptr[8] = make_constant(&ptr->base_level);
			} else {
				args_ptr[8] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1](Types::u32()), 1 } }));
			}
			if (value.level_count != VK_REMAINING_MIP_LEVELS) {
				args_ptr[9] = make_constant(&ptr->level_count);
			} else {
				args_ptr[9] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1](Types::u32()), 1 } }));
			}

			return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
			                              .type = std::span{ new std::shared_ptr<Type>[1](Types::get_builtin_image()), 1 },
			                              .construct = { .args = std::span(args_ptr, 10) } }));
		}

		Ref make_declare_buffer(Buffer value) {
			auto buf_ptr = new (new char[sizeof(Buffer)]) Buffer(value); /* rest size */
			auto args_ptr = new Ref[2];
			auto mem_ty = new std::shared_ptr<Type>[1](Types::memory(sizeof(Buffer)));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = buf_ptr, .owned = true } }));
			if (value.size != ~(0u)) {
				args_ptr[1] = make_constant(&buf_ptr->size);
			} else {
				args_ptr[1] = first(emplace_op(Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1](Types::u64()), 1 } }));
			}

			return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
			                              .type = std::span{ new std::shared_ptr<Type>[1](Types::get_builtin_buffer()), 1 },
			                              .construct = { .args = std::span(args_ptr, 2) } }));
		}

		Ref make_declare_array(std::shared_ptr<Type> type, std::span<Ref> args) {
			auto arr_ty = new std::shared_ptr<Type>[1](Types::global().emplace_type(std::shared_ptr<Type>(
			    new Type{ .kind = Type::ARRAY_TY, .size = args.size() * type->size, .array = { .T = type, .count = args.size(), .stride = type->size } })));
			auto args_ptr = new Ref[args.size() + 1];
			auto mem_ty = new std::shared_ptr<Type>[1](Types::memory(0));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = nullptr } }));
			std::copy(args.begin(), args.end(), args_ptr + 1);
			return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ arr_ty, 1 }, .construct = { .args = std::span(args_ptr, args.size() + 1) } }));
		}

		Ref make_declare_swapchain(Swapchain& bundle) {
			auto args_ptr = new Ref[2];
			auto mem_ty = new std::shared_ptr<Type>[1](Types::memory(sizeof(Swapchain*)));
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = &bundle, .owned = false } }));
			std::vector<Ref> imgs;
			for (auto i = 0; i < bundle.images.size(); i++) {
				imgs.push_back(make_declare_image(bundle.images[i]));
			}
			args_ptr[1] = make_declare_array(Types::get_builtin_image(), imgs);
			return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
			                              .type = std::span{ new std::shared_ptr<Type>[1](Types::get_builtin_swapchain()), 1 },
			                              .construct = { .args = std::span(args_ptr, 2) } }));
		}

		Ref make_extract(Ref composite, Ref index) {
			auto stripped = Type::stripped(composite.type());
			assert(stripped->kind == Type::ARRAY_TY);
			auto ty = new std::shared_ptr<Type>[1](stripped->array.T);
			return first(emplace_op(Node{ .kind = Node::EXTRACT, .type = std::span{ ty, 1 }, .extract = { .composite = composite, .index = index } }));
		}

		Ref make_extract(Ref composite, uint64_t index) {
			auto ty = new std::shared_ptr<Type>[1];
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
			auto ty = new std::shared_ptr<Type>[2](stripped, stripped);
			return first(emplace_op(
			    Node{ .kind = Node::SLICE,
			          .type = std::span{ ty, 2 },
			          .slice = { .image = image, .base_level = base_level, .level_count = level_count, .base_layer = base_layer, .layer_count = layer_count } }));
		}

		// slice splits a range into two halves
		// converge is essentially an unslice -> it returns back to before the slice was made
		// since a slice source is always a single range, converge produces a single range too
		Ref make_converge(std::span<Ref> deps, std::span<char> write) {
			auto stripped = Type::stripped(deps[0].type());
			auto ty = new std::shared_ptr<Type>[1](stripped);

			auto deps_ptr = new Ref[deps.size()];
			std::copy(deps.begin(), deps.end(), deps_ptr);
			auto rw_ptr = new bool[deps.size()];
			std::copy(write.begin(), write.end(), rw_ptr);
			return first(emplace_op(Node{ .kind = Node::CONVERGE,
			                              .type = std::span{ ty, 1 },
			                              .converge = { .diverged = std::span{ deps_ptr, deps.size() }, .write = std::span{ rw_ptr, deps.size() } } }));
		}

		Ref make_cast(std::shared_ptr<Type> dst_type, Ref src) {
			auto ty = new std::shared_ptr<Type>[1](dst_type);
			return first(emplace_op(Node{ .kind = Node::CAST, .type = std::span{ ty, 1 }, .cast = { .src = src } }));
		}

		Ref make_acquire_next_image(Ref swapchain) {
			return first(emplace_op(Node{ .kind = Node::ACQUIRE_NEXT_IMAGE,
			                              .type = std::span{ new std::shared_ptr<Type>[1](Types::get_builtin_image()), 1 },
			                              .acquire_next_image = { .swapchain = swapchain } }));
		}

		Ref make_clear_image(Ref dst, Clear cv) {
			return first(emplace_op(Node{ .kind = Node::CLEAR,
			                              .type = std::span{ new std::shared_ptr<Type>[1](Types::get_builtin_image()), 1 },
			                              .clear = { .dst = dst, .cv = new Clear(cv) } }));
		}

		std::shared_ptr<Type> make_opaque_fn_ty(std::span<std::shared_ptr<Type> const> args,
		                                        std::span<std::shared_ptr<Type> const> ret_types,
		                                        DomainFlags execute_on,
		                                        UserCallbackType callback,
		                                        std::string_view name) {
			auto arg_ptr = std::vector<std::shared_ptr<Type>>(args.size());
			std::copy(args.begin(), args.end(), arg_ptr.begin());
			auto ret_ty_ptr = std::vector<std::shared_ptr<Type>>(ret_types.size());
			std::copy(ret_types.begin(), ret_types.end(), ret_ty_ptr.begin());
			auto ty = Types::global().emplace_type(std::shared_ptr<Type>(
			    new Type{ .kind = Type::OPAQUE_FN_TY,
			              .opaque_fn = { .args = arg_ptr, .return_types = ret_ty_ptr, .execute_on = execute_on.m_mask, .callback = std::move(callback) } }));
			ty->debug_info = Types::global().allocate_type_debug_info(std::string(name));
			return ty;
		}

		Ref make_declare_fn(std::shared_ptr<Type> const fn_ty) {
			auto ty = new std::shared_ptr<Type>[1](fn_ty);
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = nullptr } }));
		}

		template<class... Refs>
		Node* make_call(Ref fn, Refs... args) {
			Ref* args_ptr = new Ref[sizeof...(args) + 1]{ fn, args... };
			decltype(Node::call) call = { .args = std::span(args_ptr, sizeof...(args) + 1) };
			Node n{};
			n.kind = Node::CALL;
			n.type = { new std::shared_ptr<Type>[fn.type()->opaque_fn.return_types.size()], fn.type()->opaque_fn.return_types.size() };
			std::copy(fn.type()->opaque_fn.return_types.begin(), fn.type()->opaque_fn.return_types.end(), n.type.data());
			n.call = call;
			return emplace_op(n);
		}

		Node* make_splice(Node* src, AcquireRelease* acq_rel, Access dst_access = Access::eNone, DomainFlagBits dst_domain = DomainFlagBits::eAny) {
			Ref* args_ptr = new Ref[src->type.size()];
			auto tys = new std::shared_ptr<Type>[src->type.size()];
			for (size_t i = 0; i < src->type.size(); i++) {
				args_ptr[i] = Ref{ src, i };
				tys[i] = Type::stripped(src->type[i]);
			}
			return emplace_op(
			    Node{ .kind = Node::SPLICE,
			          .type = std::span{ tys, src->type.size() },
			          .splice = { .src = std::span{ args_ptr, src->type.size() }, .rel_acq = acq_rel, .dst_access = dst_access, .dst_domain = dst_domain } });
		}

		Ref make_ref_splice(Ref src, AcquireRelease* acq_rel, Access dst_access = Access::eNone, DomainFlagBits dst_domain = DomainFlagBits::eAny) {
			Ref* args_ptr = new Ref[1]{ src };
			auto tys = new std::shared_ptr<Type>[1]{ src.type() };
			return first(emplace_op(
			    Node{ .kind = Node::SPLICE,
			          .type = std::span{ tys, 1 },
			          .splice = { .src = std::span{ args_ptr, 1 }, .rel_acq = acq_rel, .dst_access = dst_access, .dst_domain = dst_domain } }));
		}

		template<class T>
		Ref acquire(std::shared_ptr<Type> type, AcquireRelease* acq_rel, T value) {
			auto val_ptr = new T(value);

			auto tys = new std::shared_ptr<Type>[1]{ type };
			auto vals = new void*[1]{ val_ptr };
			return first(emplace_op(Node{ .kind = Node::SPLICE, .type = std::span{ tys, 1 }, .splice = { .rel_acq = acq_rel, .values = std::span{ vals, 1 } } }));
		}

		Node* copy_node(Node* node) {
			return emplace_op(*node);
		}

		// MATH

		Ref make_math_binary_op(Node::BinOp op, Ref a, Ref b) {
			std::shared_ptr<Type>* tys = new std::shared_ptr<Type>[1](a.type());

			return first(emplace_op(Node{ .kind = Node::MATH_BINARY, .type = std::span{ tys, 1 }, .math_binary = { .a = a, .b = b, .op = op } }));
		}
	};

	inline thread_local std::shared_ptr<IRModule> current_module = std::make_shared<IRModule>();

	struct ExtNode {
		ExtNode(Node* node, std::vector<std::shared_ptr<ExtNode>> deps) : deps(std::move(deps)) {
			acqrel = std::make_unique<AcquireRelease>();
			this->node = current_module->make_splice(node, acqrel.get());

			source_module = current_module;
		}

		ExtNode(Node* node, std::shared_ptr<ExtNode> dep) {
			acqrel = std::make_unique<AcquireRelease>();
			this->node = current_module->make_splice(node, acqrel.get());

			deps.push_back(std::move(dep));

			source_module = current_module;
		}

		ExtNode(Ref ref, std::shared_ptr<ExtNode> dep, Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eAny) {
			acqrel = std::make_unique<AcquireRelease>();
			this->node = current_module->make_ref_splice(ref, acqrel.get(), access, domain).node;

			deps.push_back(std::move(dep));

			source_module = current_module;
		}
		// for releases
		ExtNode(Node* node, std::shared_ptr<ExtNode> dep, Access access, DomainFlagBits domain) {
			acqrel = std::make_unique<AcquireRelease>();
			this->node = current_module->make_splice(node, acqrel.get(), access, domain);

			deps.push_back(std::move(dep));

			source_module = current_module;
		}

		// for acquires - adopt the node
		ExtNode(Node* node, ResourceUse use) : node(node) {
			acqrel = std::make_unique<AcquireRelease>();
			acqrel->status = Signal::Status::eHostAvailable;
			acqrel->last_use.resize(1);
			acqrel->last_use[0] = use;

			node->splice.rel_acq = acqrel.get();

			source_module = current_module;
		}

		~ExtNode() {
			if (acqrel) {
				assert(node->kind == Node::SPLICE);
				node->splice.rel_acq = nullptr;
				for (auto i = 0; i < node->splice.values.size(); i++) {
					auto& v = node->splice.values[i];
					if (node->type[i]->hash_value == Types::global().builtin_buffer) {
						delete (Buffer*)v;
					} else {
						delete (ImageAttachment*)v;
					}
				}
				delete node->splice.values.data();

				source_module->potential_garbage.emplace(node, 0);
			}
		}

		ExtNode(ExtNode&& o) = delete;
		ExtNode& operator=(ExtNode&& o) = delete;

		Node* get_node() {
			assert(node->kind == Node::SPLICE);
			return node;
		}

		void mutate(Node* new_node) {
			current_module->garbage.push_back(node);
			assert(node->kind == Node::SPLICE);
			node->splice.rel_acq = nullptr;
			node = current_module->make_splice(new_node, acqrel.get());
		}

		std::unique_ptr<AcquireRelease> acqrel;
		std::vector<std::shared_ptr<ExtNode>> deps;
		std::shared_ptr<IRModule> source_module;

	private:
		Node* node;
	};

	struct ExtRef {
		ExtRef(std::shared_ptr<ExtNode> node, Ref ref) : node(node), index(ref.index) {}

		std::shared_ptr<ExtNode> node;
		size_t index;
	};
} // namespace vuk
