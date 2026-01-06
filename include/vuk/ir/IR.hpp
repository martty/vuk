#pragma once

#include "vuk/ImageAttachment.hpp"
#include "vuk/RelSpan.hpp"
#include "vuk/ResourceUse.hpp"
#include "vuk/runtime/vk/Allocation.hpp"
#include "vuk/runtime/vk/VkSwapchain.hpp" //TODO: leaking vk
#include "vuk/ShortAlloc.hpp"
#include "vuk/SourceLocation.hpp"
#include "vuk/SyncPoint.hpp"
#include "vuk/Types.hpp"

#include <atomic>
#include <deque>
#include <function2/function2.hpp>
#include <optional>
#include <plf_colony.h>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

// #define VUK_GARBAGE_SAN

namespace vuk {
	struct IRModule;

	struct TypeDebugInfo {
		std::string name;
	};

	using UserCallbackType = fu2::unique_function<void(CommandBuffer&, std::span<void*>, std::span<void*>, std::span<void*>)>;

	struct Type {
		enum TypeKind {
			VOID_TY = 0,
			MEMORY_TY = 1,
			INTEGER_TY,
			FLOAT_TY,
			POINTER_TY,
			COMPOSITE_TY,
			ARRAY_TY,
			UNION_TY,
			IMBUED_TY,
			ALIASED_TY,
			OPAQUE_FN_TY,
			SHADER_FN_TY,
			ENUM_TY,
			ENUM_VALUE_TY,
			IMAGE_TY,
			OPAQUE_TY
		} kind;
		enum Tags { TAG_IMAGE = 3, TAG_SWAPCHAIN = 4 };
		size_t size = ~0ULL;

		TypeDebugInfo debug_info;

		std::vector<std::shared_ptr<Type>> child_types;
		std::vector<size_t> offsets;                // for now only useful for composites
		std::unique_ptr<UserCallbackType> callback; // only useful for user CBs
		std::vector<const char*> member_names;
		void (*format_to)(void* value, std::string&) = nullptr;

		union {
			struct {
				uint32_t width;
			} scalar;
			struct {
				std::shared_ptr<Type>* T;
				Access access;
			} imbued;
			struct {
				std::shared_ptr<Type>* T;
				size_t ref_idx;
			} aliased;
			struct {
				std::span<std::shared_ptr<Type>> args;
				std::span<std::shared_ptr<Type>> return_types;
				size_t hash_code;
				DomainFlags execute_on;
			} opaque_fn;
			struct {
				void* shader;
				std::span<std::shared_ptr<Type>> args;
				std::span<std::shared_ptr<Type>> return_types;
				DomainFlags execute_on;
			} shader_fn;
			struct {
				std::shared_ptr<Type>* T;
			} pointer;
			struct {
				std::shared_ptr<Type>* T;
				size_t count;
				size_t stride;
			} array;
			struct {
				std::span<std::shared_ptr<Type>> types;
				size_t tag;
				void (*construct)(void* dst, std::span<void*> args) = nullptr;
				void* (*get)(void* value, size_t index) = nullptr;
				bool (*is_default)(void* value, size_t index) = nullptr;
				void (*destroy)(void* dst) = nullptr;
				void (*synchronize)(void*, struct SyncHelper&) = nullptr;
			} composite;
			struct {
				size_t tag;
			} enumt;
			struct {
				std::shared_ptr<Type>* enum_type;
				uint64_t value;
			} enum_value;
			struct {
				size_t tag;
			} opaque;
		};

		~Type() {}

		static std::shared_ptr<Type> stripped(std::shared_ptr<Type> t) {
			switch (t->kind) {
			case IMBUED_TY:
				return stripped(*t->imbued.T);
			case ALIASED_TY:
				return stripped(*t->aliased.T);
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

		static Hash hash_scalar(Type::TypeKind kind, size_t width) {
			Hash v = (Hash)kind;
			hash_combine_direct(v, width);
			return v;
		}

		static Hash hash(Type const* t) {
			Hash v = (Hash)t->kind;
			switch (t->kind) {
			case VOID_TY:
				return v;
			case IMBUED_TY:
				hash_combine_direct(v, Type::hash(t->imbued.T->get()));
				hash_combine_direct(v, (uint32_t)t->imbued.access);
				return v;
			case ALIASED_TY:
				hash_combine_direct(v, Type::hash(t->aliased.T->get()));
				hash_combine_direct(v, (uint32_t)t->aliased.ref_idx);
				return v;
			case MEMORY_TY:
				hash_combine_direct(v, (uint32_t)t->size);
				return v;
			case INTEGER_TY:
			case FLOAT_TY:
				hash_combine_direct(v, t->scalar.width);
				return v;
			case ARRAY_TY:
				hash_combine_direct(v, Type::hash(t->array.T->get()));
				hash_combine_direct(v, (uint32_t)t->array.count);
				return v;
			case UNION_TY:
			case COMPOSITE_TY: {
				for (int i = 0; i < t->composite.types.size(); i++) {
					hash_combine_direct(v, Type::hash(t->composite.types[i].get()));
				}
				hash_combine_direct(v, (uint32_t)t->composite.tag);
				return v;
			}
			case OPAQUE_FN_TY:
				hash_combine_direct(v, (uintptr_t)t->opaque_fn.hash_code >> 32);
				hash_combine_direct(v, (uintptr_t)t->opaque_fn.hash_code & 0xffffffff);
				return v;
			case SHADER_FN_TY:
				hash_combine_direct(v, (uintptr_t)t->shader_fn.shader >> 32);
				hash_combine_direct(v, (uintptr_t)t->shader_fn.shader & 0xffffffff);
				return v;
			case POINTER_TY:
				hash_combine_direct(v, Type::hash(t->pointer.T->get()));
				return v;
			case IMAGE_TY:
				hash_combine_direct(v, Type::hash(t->pointer.T->get()));
				return v;
			case ENUM_TY:
				hash_combine_direct(v, (uint32_t)t->enumt.tag);
				return v;
			case ENUM_VALUE_TY:
				hash_combine_direct(v, Type::hash(t->enum_value.enum_type->get()));
				hash_combine_direct(v, (uint32_t)(t->enum_value.value >> 32));
				hash_combine_direct(v, (uint32_t)(t->enum_value.value & 0xffffffff));
				return v;
			case OPAQUE_TY:
				hash_combine_direct(v, (uint32_t)t->opaque.tag);
				return v;
			}
			assert(0);
			return v;
		}

		bool is_bufferlike_view() const {
			return kind == Type::COMPOSITE_TY && composite.types.size() == 2 && composite.types[0]->kind == Type::POINTER_TY &&
			       composite.types[1]->kind == Type::INTEGER_TY && composite.types[1]->scalar.width == 64;
		}

		bool is_imageview() const {
			return kind == Type::POINTER_TY && pointer.T->get()->kind == Type::ENUM_VALUE_TY;
		}

		bool is_synchronized() {
			if (kind == Type::COMPOSITE_TY && composite.synchronize != nullptr) {
				return true;
			}
			if (kind == Type::IMBUED_TY) {
				return imbued.T->get()->is_synchronized();
			}
			if (kind == Type::ALIASED_TY) {
				return aliased.T->get()->is_synchronized();
			}
			if (kind == Type::UNION_TY) {
				return true;
			}
			if (is_bufferlike_view() || is_imageview()) {
				return true;
			}
			return false;
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
			case eTessellationRead:
				return "TessR";
			case eTessellationSampled:
				return "TessS";
			case vuk::eTessellationUniformRead:
				return "TessU";
			case eCopyRead:
				return "CopyR";
			case eCopyWrite:
				return "CopyW";
			case eCopyRW:
				return "CopyRW";
			case eBlitRead:
				return "BlitR";
			case eBlitWrite:
				return "BlitW";
			case eBlitRW:
				return "BlitRW";
			case eResolveRead:
				return "ResolvR";
			case eResolveWrite:
				return "ResolvW";
			case eResolveRW:
				return "ResolvRW";
			default:
				return "<multiple>";
			}
		}

		static std::string to_string(Type* t) {
			switch (t->kind) {
			case VOID_TY:
				return "void";
			case IMBUED_TY:
				return to_string(t->imbued.T->get()) + std::string(":") + std::string(to_sv(t->imbued.access));
			case ALIASED_TY:
				return to_string(t->aliased.T->get()) + std::string("@") + std::to_string(t->aliased.ref_idx);
			case MEMORY_TY:
				return "mem";
			case INTEGER_TY:
				return t->scalar.width == 32 ? "i32" : "i64";
			case FLOAT_TY:
				return t->scalar.width == 32 ? "f32" : "f64";
			case ARRAY_TY:
				return to_string(t->array.T->get()) + "[" + std::to_string(t->array.count) + "]";
			case COMPOSITE_TY:
				if (!t->debug_info.name.empty()) {
					return std::string(t->debug_info.name);
				}
				return "composite:" + std::to_string(t->composite.tag);
			case UNION_TY:
				if (!t->debug_info.name.empty()) {
					return std::string(t->debug_info.name);
				}
				return "union:" + std::to_string(t->composite.tag);
			case ENUM_TY:
				if (!t->debug_info.name.empty()) {
					return std::string(t->debug_info.name);
				}
				return "enum:" + std::to_string(t->enumt.tag);
			case ENUM_VALUE_TY: {
				std::string result;
				if (t->enum_value.enum_type->get()->format_to) {
					std::string formatted;
					t->enum_value.enum_type->get()->format_to((void*)&t->enum_value.value, formatted);
					result += formatted;
				} else {
					result += std::to_string(t->enum_value.value);
				}
				return result;
			}
			case OPAQUE_FN_TY:
				return "ofn";
			case SHADER_FN_TY:
				return "sfn";
			case POINTER_TY:
				return to_string(t->pointer.T->get()) + "*";
			case OPAQUE_TY:
				return "opaque:" + std::to_string(t->opaque.tag);
			case IMAGE_TY:
				return "image";
			default:
				assert(0);
				return "?";
			}
		}
	};

	template<class Type, Access acc, class UniqueT>
	size_t Arg<Type, acc, UniqueT>::size() const noexcept
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
		std::vector<std::string> result_names;
		std::span<vuk::source_location> trace;
	};

	// struct describing use chains
	struct ChainLink {
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
			SLICE, /* SLICED REST ORIGINAL */
			CONVERGE,
			IMPORT,
			CALL,
			CLEAR,
			ACQUIRE,
			RELEASE,
			ACQUIRE_NEXT_IMAGE,
			USE,
			LOGICAL_COPY,
			SET,
			CAST,
			MATH_BINARY,
			COMPILE_PIPELINE,
			ALLOCATE,
			GET_ALLOCATION_SIZE,
			GET_CI,
			GARBAGE,
			NODE_KIND_MAX
		} kind;
		uint8_t flag = 0;
		std::span<std::shared_ptr<Type>> type;
		NodeDebugInfo* debug_info = nullptr;
		SchedulingInfo* scheduling_info = nullptr;
		ChainLink* links = nullptr;
		ExecutionInfo* execution_info = nullptr;
		struct ScheduledItem* scheduled_item = nullptr;
		size_t index;
		AcquireRelease* rel_acq = nullptr;
		bool held = false;
		DomainFlags compute_class = DomainFlagBits::eDevice;

		template<uint8_t c>
		struct Fixed {
			static_assert(c <= MAX_ARGS);
			uint8_t arg_count = c;
		};

		struct Variable {
			uint8_t arg_count = (uint8_t)~0u;
		};

		enum NamedAxis : uint8_t {
			FIELD = 254,
			MIP = 253,
			LAYER = 252,
			COMPONENT = 251,
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
			} construct;
			struct : Fixed<3> {
				Ref src;
				Ref start;
				Ref count;
				uint8_t axis;
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
				std::span<void*> values;
			} acquire;
			struct : Variable {
				std::span<Ref> src;
				Access dst_access;
				DomainFlagBits dst_domain;
			} release;
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
				Ref src;
				std::optional<Allocator> allocator;
			} allocate;
			struct : Fixed<1> {
				Ref src;
				Access access;
			} use;
			struct : Fixed<1> {
				Ref src;
			} logical_copy;
			struct : Fixed<2> {
				Ref dst;
				Ref value;
				int index;
				bool set_on_allocate = false;
			} set;
			struct : Fixed<1> {
				Ref src;
			} compile_pipeline;
			struct : Fixed<1> {
				Ref ptr;
			} get_allocation_size;
			struct : Fixed<1> {
				Ref src;
			} get_ci;
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

		static std::string_view kind_to_sv(Node::Kind kind) {
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
			case MATH_BINARY:
				return "math_b";
			case SLICE:
				return "slice";
			case CONVERGE:
				return "converge";
			case CLEAR:
				return "clear";
			case CAST:
				return "cast";
			case GARBAGE:
				return "garbage";
			case RELEASE:
				return "release";
			case ACQUIRE:
				return "acquire";
			case USE:
				return "use";
			case SET:
				return "set";
			case LOGICAL_COPY:
				return "lcopy";
			case GET_ALLOCATION_SIZE:
				return "get_allocation_size";
			case GET_CI:
				return "get_ci";
			case COMPILE_PIPELINE:
				return "compile_pipeline";
			case ALLOCATE:
				return "allocate";
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
		assert(ref.node->kind == Node::CONSTANT);
		return *reinterpret_cast<T*>(ref.node->constant.value);
	}

	inline void* constant(Ref ref) {
		assert(ref.node->kind == Node::CONSTANT);
		return ref.node->constant.value;
	}

	struct CannotBeConstantEvaluated : Exception {
		CannotBeConstantEvaluated(Ref ref) : ref(ref) {}

		Ref ref;

		void throw_this() override {
			throw *this;
		}
	};

	enum class RW { eRead, eWrite };

	struct RefOrValue {
		Ref ref;
		std::unique_ptr<char[]> owned_value;
		void* value;
		bool is_ref;

		static RefOrValue from_ref(Ref r) {
			return { r, nullptr, nullptr, true };
		}
		static RefOrValue from_value(void* v, Ref r = {}) {
			return { r, nullptr, v, false };
		}
		static RefOrValue adopt_value(void* v, Ref r = {}) {
			return { r, std::unique_ptr<char[]>(static_cast<char*>(v)), v, false };
		}
	};

	template<class T, size_t size>
	struct InlineArena {
		std::unique_ptr<InlineArena> next;
		std::byte arena[size];
		std::byte* base;
		std::byte* cur;
		std::vector<std::unique_ptr<char[]>> large_allocs;

		InlineArena() {
			base = cur = arena;
		}

		void reset() {
			next.reset();
			base = cur = arena;
		}

		void* ensure_space(size_t ns) {
			if (ns > size) {
				auto& alloc = large_allocs.emplace_back(new char[ns]);
				return static_cast<void*>(alloc.get());
			}

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

		template<class U>
		U* emplace(U v) {
			return new (ensure_space(sizeof(U))) U(std::move(v));
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
		IRModule() : op_arena(/**/), module_id(module_id_counter++) {
			// prepopulate builtin type hashes
			types.get_builtin_sampler();
			types.get_builtin_sampled_image();
			types.get_builtin_swapchain();
		}

		plf::colony<Node /*, inline_alloc<Node, 4 * 1024>*/> op_arena;
		std::vector<Node*> garbage;
		size_t node_counter = 0;
		size_t link_frontier = 0;
		size_t module_id = 0;
		inline static std::atomic<size_t> module_id_counter;

		struct Types {
			std::unordered_map<Type::Hash, std::weak_ptr<Type>> type_map;
			plf::colony<UserCallbackType> ucbs;

			Type::Hash builtin_swapchain = 0;
			Type::Hash builtin_sampler = 0;
			Type::Hash builtin_sampled_image = 0;

			size_t union_tag_type_counter = 0;

			// TYPES
			std::shared_ptr<Type> make_imbued_ty(std::shared_ptr<Type> ty, Access access) {
				auto t = new Type{ .kind = Type::IMBUED_TY, .size = ty->size, .imbued = { .access = access } };
				t->imbued.T = &t->child_types.emplace_back(ty);
				return emplace_type(std::shared_ptr<Type>(t));
			}

			std::shared_ptr<Type> make_aliased_ty(std::shared_ptr<Type> ty, size_t ref_idx) {
				auto t = new Type{ .kind = Type::ALIASED_TY, .size = ty->size, .aliased = { .ref_idx = ref_idx } };
				t->imbued.T = &t->child_types.emplace_back(ty);
				return emplace_type(std::shared_ptr<Type>(t));
			}

			std::shared_ptr<Type> make_array_ty(std::shared_ptr<Type> ty, size_t count) {
				auto t = new Type{ .kind = Type::ARRAY_TY, .size = count * ty->size, .array = { .count = count, .stride = ty->size } };
				t->array.T = &t->child_types.emplace_back(ty);
				return emplace_type(std::shared_ptr<Type>(t));
			}

			std::shared_ptr<Type> make_pointer_ty(std::shared_ptr<Type> ty) {
				auto t = new Type{ .kind = Type::POINTER_TY, .size = sizeof(uint64_t), .pointer = {} };
				t->pointer.T = &t->child_types.emplace_back(ty);
				return emplace_type(std::shared_ptr<Type>(t));
			}

			// an opaque pointer I guess
			std::shared_ptr<Type> make_image_ty(std::shared_ptr<Type> ty) {
				auto t = new Type{ .kind = Type::IMAGE_TY, .size = sizeof(uint64_t), .pointer = {} };
				t->pointer.T = &t->child_types.emplace_back(ty);
				return emplace_type(std::shared_ptr<Type>(t));
			}

			std::shared_ptr<Type> make_union_ty(std::vector<std::shared_ptr<Type>> types) {
				std::vector<size_t> offsets;
				size_t offset = 0;
				for (auto& t : types) {
					offsets.push_back(offset);
					offset += t->size;
				}
				auto union_type = emplace_type(std::shared_ptr<Type>(
				    new Type{ .kind = Type::UNION_TY, .size = offset, .offsets = offsets, .composite = { .types = types, .tag = union_tag_type_counter++ } }));
				union_type->child_types = std::move(types);
				return union_type;
			}

			std::shared_ptr<Type> make_opaque_fn_ty(std::span<std::shared_ptr<Type> const> args,
			                                        std::span<std::shared_ptr<Type> const> ret_types,
			                                        DomainFlags execute_on,
			                                        size_t hash_code,
			                                        UserCallbackType callback,
			                                        std::string_view name) {
				auto arg_ptr_ret_ty_ptr = std::vector<std::shared_ptr<Type>>(args.size() + ret_types.size());
				auto it = std::copy(args.begin(), args.end(), arg_ptr_ret_ty_ptr.begin());
				std::copy(ret_types.begin(), ret_types.end(), it);
				auto t = new Type{ .kind = Type::OPAQUE_FN_TY,
					                 .opaque_fn = { .args = std::span{ arg_ptr_ret_ty_ptr.data(), args.size() },
					                                .return_types = std::span{ arg_ptr_ret_ty_ptr.data() + args.size(), ret_types.size() },
					                                .hash_code = hash_code,
					                                .execute_on = execute_on } };
				t->callback = std::make_unique<UserCallbackType>(std::move(callback));
				t->child_types = std::move(arg_ptr_ret_ty_ptr);
				t->debug_info = allocate_type_debug_info(std::string(name));
				auto tc = std::shared_ptr<Type>(t);
				emplace_type(tc);
				return tc; // we don't dedupe the outer type, only the inner type - this means all opaque_fn_tys will be unique
			}

			std::shared_ptr<Type> make_shader_fn_ty(std::span<std::shared_ptr<Type> const> args,
			                                        std::span<std::shared_ptr<Type> const> ret_types,
			                                        DomainFlags execute_on,
			                                        void* shader,
			                                        std::string_view name) {
				auto arg_ptr_ret_ty_ptr = std::vector<std::shared_ptr<Type>>(args.size() + ret_types.size());
				auto it = std::copy(args.begin(), args.end(), arg_ptr_ret_ty_ptr.begin());
				std::copy(ret_types.begin(), ret_types.end(), it);
				auto t = new Type{ .kind = Type::SHADER_FN_TY,
					                 .shader_fn = { .shader = shader,
					                                .args = std::span{ arg_ptr_ret_ty_ptr.data(), args.size() },
					                                .return_types = std::span{ arg_ptr_ret_ty_ptr.data() + args.size(), ret_types.size() },
					                                .execute_on = execute_on } };
				t->child_types = std::move(arg_ptr_ret_ty_ptr);
				t->debug_info = allocate_type_debug_info(std::string(name));
				return emplace_type(std::shared_ptr<Type>(t));
			}

			std::shared_ptr<Type> make_void_ty() {
				auto hash = 1;
				auto it = type_map.find(hash);
				if (it != type_map.end()) {
					if (auto ty = it->second.lock()) {
						return ty;
					}
				}

				return emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::VOID_TY, .size = 0 }));
			}

			std::shared_ptr<Type> make_scalar_ty(Type::TypeKind kind, uint32_t bit_width) {
				auto hash = Type::hash_scalar(kind, bit_width);
				auto it = type_map.find(hash);
				if (it != type_map.end()) {
					if (auto ty = it->second.lock()) {
						return ty;
					}
				}

				return emplace_type(std::shared_ptr<Type>(new Type{ .kind = kind, .size = bit_width / 8, .scalar = { .width = bit_width } }));
			}

			std::shared_ptr<Type> make_enum_ty() {
				auto t = new Type{ .kind = Type::ENUM_TY, .size = 4, .enumt = { .tag = 0 } };
				return emplace_type(std::shared_ptr<Type>(t));
			};

			std::shared_ptr<Type> make_enum_ty(size_t tag, void (*format_to)(void*, std::string&) = nullptr, size_t size = 4) {
				auto t = new Type{ .kind = Type::ENUM_TY, .size = size, .format_to = format_to, .enumt = { .tag = tag } };
				return emplace_type(std::shared_ptr<Type>(t));
			}

			std::shared_ptr<Type> make_enum_value_ty(std::shared_ptr<Type> enum_type, uint64_t value) {
				assert(enum_type->kind == Type::ENUM_TY);
				auto t = new Type{ .kind = Type::ENUM_VALUE_TY, .size = enum_type->size, .enum_value = { .value = value } };
				t->enum_value.enum_type = &t->child_types.emplace_back(enum_type);
				return emplace_type(std::shared_ptr<Type>(t));
			}

			std::shared_ptr<Type> make_opaque_ty(size_t tag, size_t size = sizeof(uint32_t)) {
				auto t = new Type{ .kind = Type::OPAQUE_TY, .size = size, .opaque = { .tag = tag } };
				return emplace_type(std::shared_ptr<Type>(t));
			}

			std::shared_ptr<Type> u32() {
				return make_scalar_ty(Type::INTEGER_TY, 32);
			}

			std::shared_ptr<Type> u64() {
				return make_scalar_ty(Type::INTEGER_TY, 64);
			}

			std::shared_ptr<Type> memory(size_t size) {
				Type ty{ .kind = Type::MEMORY_TY, .size = size };
				auto it = type_map.find(Type::hash(&ty));
				if (it != type_map.end()) {
					if (auto ty = it->second.lock()) {
						return ty;
					}
				}
				return emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::MEMORY_TY, .size = size }));
			}

			std::shared_ptr<Type> make_imageview_ty();
			std::shared_ptr<Type> make_imageview_ty(std::shared_ptr<Type>);

			std::shared_ptr<Type> get_builtin_swapchain() {
				if (builtin_swapchain) {
					auto it = type_map.find(builtin_swapchain);
					if (it != type_map.end()) {
						if (auto ty = it->second.lock()) {
							return ty;
						}
					}
				}
				auto arr_ty = make_array_ty(make_imageview_ty(), 16);
				auto swp_ = std::vector<std::shared_ptr<Type>>{ arr_ty };
				auto offsets = std::vector<size_t>{ 0 };

				auto swapchain_type = emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
				                                                                   .size = sizeof(Swapchain*),
				                                                                   .debug_info = allocate_type_debug_info("swapchain"),
				                                                                   .offsets = offsets,
				                                                                   .composite = { .types = swp_, .tag = Type::TAG_SWAPCHAIN } }));
				builtin_swapchain = Type::hash(swapchain_type.get());
				return swapchain_type;
			}

			std::shared_ptr<Type> get_builtin_sampler() {
				if (builtin_sampler) {
					auto it = type_map.find(builtin_sampler);
					if (it != type_map.end()) {
						if (auto ty = it->second.lock()) {
							return ty;
						}
					}
				}
				auto sampler_type = emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
				                                                                 .size = sizeof(SamplerCreateInfo),
				                                                                 .debug_info = allocate_type_debug_info("sampler"),
				                                                                 .offsets = {},
				                                                                 .composite = { .types = {}, .tag = 3 } }));
				builtin_sampler = Type::hash(sampler_type.get());
				return sampler_type;
			}

			std::shared_ptr<Type> get_builtin_sampled_image() {
				if (builtin_sampled_image) {
					auto it = type_map.find(builtin_sampled_image);
					if (it != type_map.end()) {
						if (auto ty = it->second.lock()) {
							return ty;
						}
					}
				}
				auto sampled_image_type = emplace_type(std::shared_ptr<Type>(new Type{ .kind = Type::COMPOSITE_TY,
				                                                                       .size = sizeof(SampledImage),
				                                                                       .debug_info = allocate_type_debug_info("sampled_image"),
				                                                                       .offsets = {},
				                                                                       .composite = { .types = {}, .tag = 4 } }));
				builtin_sampled_image = Type::hash(sampled_image_type.get());
				return sampled_image_type;
			}

			std::shared_ptr<Type> emplace_type(std::shared_ptr<Type> t) {
				auto unify_type = [&](std::shared_ptr<Type>& t) {
					auto th = Type::hash(t.get());
					auto [v, succ] = type_map.try_emplace(th, t);
					if (succ) {
						t->hash_value = th;
					} else if (!v->second.lock()) {
						type_map[th] = t;
					} else {
						t = v->second.lock();
					}
					t->hash_value = th;
				};

				if (t->kind == Type::ALIASED_TY) {
					assert((*t->aliased.T)->kind != Type::ALIASED_TY);
					unify_type(*t->aliased.T);
				} else if (t->kind == Type::IMBUED_TY) {
					unify_type(*t->imbued.T);
				} else if (t->kind == Type::ARRAY_TY) {
					unify_type(*t->array.T);
				} else if (t->kind == Type::POINTER_TY) {
					unify_type(*t->pointer.T);
				} else if (t->kind == Type::ENUM_VALUE_TY) {
					unify_type(*t->enum_value.enum_type);
				} else if (t->kind == Type::COMPOSITE_TY) {
					for (auto& elem_ty : t->child_types) {
						unify_type(elem_ty);
					}
					t->composite.types = t->child_types;
				}
				unify_type(t);

				return t;
			}

			TypeDebugInfo allocate_type_debug_info(std::string name) {
				return TypeDebugInfo{ name };
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

			// TODO: PAV: this changes
			void destroy(Type* t, void* v) {
				if (t->hash_value == builtin_sampled_image) {
					std::destroy_at<SampledImage>((SampledImage*)v);
				} else if (t->hash_value == builtin_sampler) {
					std::destroy_at<SamplerCreateInfo>((SamplerCreateInfo*)v);
				} else if (t->hash_value == builtin_swapchain) {
					std::destroy_at<Swapchain*>((Swapchain**)v);
				} else if (t->kind == Type::COMPOSITE_TY) {
					t->composite.destroy(v);
				} else if (t->kind == Type::INTEGER_TY) {
					// nothing to do
				} else if (t->kind == Type::MEMORY_TY) {
					// nothing to do
				} else if (t->kind == Type::FLOAT_TY) {
					// nothing to do
				} else if (t->kind == Type::POINTER_TY) {
					// nothing to do
				} else if (t->kind == Type::VOID_TY) {
					// nothing to do
				} else if (t->kind == Type::ENUM_TY) {
					// nothing to do - enums are trivially destructible
				} else if (t->kind == Type::ENUM_VALUE_TY) {
					// nothing to do - enum values are trivially destructible
				} else if (t->kind == Type::IMBUED_TY) {
					destroy(t->imbued.T->get(), v);
				} else if (t->kind == Type::ALIASED_TY) {
					destroy(t->aliased.T->get(), v);
				} else if (t->kind == Type::OPAQUE_TY) {
					// nothing to do - opaque types don't own their values
				} else if (t->kind == Type::ARRAY_TY || t->kind == Type::UNION_TY || t->kind == Type::IMAGE_TY) {
					// currently arrays and unions don't own their values
					/* auto cv = (char*)v;
					for (auto i = 0; i < t->array.count; i++) {
					  destroy(t->array.T->get(), cv);
					  cv += t->array.stride;
					}*/
				} else {
					assert(0);
				}
				delete[] (std::byte*)v;
			}
		} types;

		Node* emplace_op(Node v) {
			v.index = module_id << 32 | node_counter++;
			return &*op_arena.emplace(std::move(v));
		}

		void name_output(Ref ref, std::string_view name) {
			auto node = ref.node;
			if (!node->debug_info) {
				node->debug_info = new NodeDebugInfo;
			}
			auto& names = ref.node->debug_info->result_names;
			if (names.size() <= ref.index) {
				names.resize(ref.index + 1);
			}
			names[ref.index] = name;
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
			if (node->debug_info->trace.data()) {
				delete[] node->debug_info->trace.data();
			}
			node->debug_info->trace = std::span(new vuk::source_location[cnt], cnt);
			p = &loc;
			cnt = 0;
			do {
				node->debug_info->trace[cnt] = p->location;
				cnt++;
				p = p->parent;
			} while (p != nullptr);
		}

		std::optional<plf::colony<Node>::iterator> destroy_node(Node* node) {
			delete node->rel_acq;
			switch (node->kind) {
			case Node::CONSTANT: {
				if (node->constant.owned) {
					delete[] (char*)node->constant.value;
				}
				break;
			}
			case Node::ACQUIRE: {
				for (auto i = 0; i < node->acquire.values.size(); i++) {
					auto& v = node->acquire.values[i];
					types.destroy(Type::stripped(node->type[i]).get(), v);
				}
				delete[] node->acquire.values.data();
				break;
			}
			default: // nothing extra to be done here
				break;
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

				delete node->debug_info;
			}

			auto it = op_arena.get_iterator(node);
			if (it != op_arena.end()) {
#ifdef VUK_GARBAGE_SAN
				node->kind = Node::GARBAGE;
				node->generic_node.arg_count = 0;
				node->type = {};

#else
				return op_arena.erase(it);
#endif
			} else {
				node->kind = Node::GARBAGE;
				node->generic_node.arg_count = 0;
				node->type = {};
			}
			return {};
		}

		// OPS

		Ref make_constant(std::shared_ptr<Type> type, void* value) {
			std::shared_ptr<Type>* ty = new std::shared_ptr<Type>[1]{ type };
			auto value_ptr = new char[type->size];
			memcpy(value_ptr, value, type->size);
			return first(emplace_op(Node{
			    .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .compute_class = DomainFlagBits::eConstant, .constant = { .value = value_ptr, .owned = true } }));
		}

		template<class T>
		Ref make_constant(T value) {
			std::shared_ptr<Type>* ty;
			if constexpr (std::is_same_v<T, uint64_t>) {
				ty = new std::shared_ptr<Type>[1]{ types.u64() };
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				ty = new std::shared_ptr<Type>[1]{ types.u32() };
			} else {
				ty = new std::shared_ptr<Type>[1]{ types.memory(sizeof(T)) };
			}
			return first(emplace_op(Node{ .kind = Node::CONSTANT,
			                              .type = std::span{ ty, 1 },
			                              .compute_class = DomainFlagBits::eConstant,
			                              .constant = { .value = new (new char[sizeof(T)]) T(value), .owned = true } }));
		}

		template<class T>
		Ref make_constant(T* value) {
			std::shared_ptr<Type>* ty;
			if constexpr (std::is_same_v<T, uint64_t>) {
				ty = new std::shared_ptr<Type>[1]{ types.u64() };
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				ty = new std::shared_ptr<Type>[1]{ types.u32() };
			} else {
				ty = new std::shared_ptr<Type>[1]{ types.memory(sizeof(T)) };
			}
			return first(emplace_op(Node{
			    .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .compute_class = DomainFlagBits::eConstant, .constant = { .value = value, .owned = false } }));
		}

		void set_value(Ref ref, Ref value) {
			emplace_op(Node{ .kind = Node::SET, .set = { .dst = ref, .value = value, .index = -1 } });
		}

		void set_value(Ref ref, size_t index, Ref value) {
			emplace_op(Node{ .kind = Node::SET, .set = { .dst = ref, .value = value, .index = (int)index } });
		}

		void set_value_on_allocate_src(Ref ref, size_t index, Ref value) {
			emplace_op(Node{ .kind = Node::SET, .set = { .dst = ref, .value = value, .index = (int)index, .set_on_allocate = true } });
		}

		template<class T>
		void set_value(Ref ref, size_t index, T value) {
			set_value(ref, index, make_constant(value));
		}

		template<class T>
		void set_value_on_allocate_src(Ref ref, size_t index, T value) {
			set_value_on_allocate_src(ref, index, make_constant(value));
		}

		Ref make_placeholder(std::shared_ptr<Type> type) {
			return first(emplace_op(
			    Node{ .kind = Node::PLACEHOLDER, .type = std::span{ new std::shared_ptr<Type>[1]{ type }, 1 }, .compute_class = DomainFlagBits::ePlaceholder }));
		}

		Ref make_declare_array(std::shared_ptr<Type> type, std::span<Ref> args) {
			auto arr_ty = new std::shared_ptr<Type>[1]{ types.make_array_ty(type, args.size()) };
			auto args_ptr = new Ref[args.size() + 1];
			auto mem_ty = new std::shared_ptr<Type>[1]{ types.memory(0) };
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = nullptr } }));
			std::copy(args.begin(), args.end(), args_ptr + 1);
			return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ arr_ty, 1 }, .construct = { .args = std::span(args_ptr, args.size() + 1) } }));
		}

		Ref make_declare_union(std::span<Ref> args) {
			std::vector<std::shared_ptr<Type>> child_types;
			for (auto& arg : args) {
				child_types.push_back(Type::stripped(arg.type()));
			}
			auto union_ty = new std::shared_ptr<Type>[1]{ types.make_union_ty(std::move(child_types)) };
			auto args_ptr = new Ref[args.size() + 1];
			auto mem_ty = new std::shared_ptr<Type>[1]{ types.memory(0) };
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = nullptr } }));
			std::copy(args.begin(), args.end(), args_ptr + 1);
			return first(
			    emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ union_ty, 1 }, .construct = { .args = std::span(args_ptr, args.size() + 1) } }));
		}

		Ref make_declare_swapchain(Swapchain& bundle) {
			auto swpptr = new (new char[sizeof(Swapchain*)]) void*(&bundle);
			auto args_ptr = new Ref[2];
			auto mem_ty = new std::shared_ptr<Type>[1]{ types.memory(sizeof(Swapchain*)) };
			args_ptr[0] = first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = swpptr, .owned = true } }));
			std::vector<Ref> imgs;
			for (auto i = 0; i < bundle.images.size(); i++) {
				imgs.push_back(make_constant(types.make_imageview_ty(), &bundle.images[i]));
			}
			args_ptr[1] = make_declare_array(types.make_imageview_ty(), imgs);
			return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
			                              .type = std::span{ new std::shared_ptr<Type>[1]{ types.get_builtin_swapchain() }, 1 },
			                              .construct = { .args = std::span(args_ptr, 2) } }));
		}

		Ref make_sampled_image(Ref image, Ref sampler) {
			auto args_ptr = new Ref[3]{ make_constant(0), image, sampler };
			return first(emplace_op(Node{ .kind = Node::CONSTRUCT,
			                              .type = std::span{ new std::shared_ptr<Type>[1]{ types.get_builtin_sampled_image() }, 1 },
			                              .construct = { .args = std::span(args_ptr, 3) } }));
		}

		Ref make_construct(std::shared_ptr<Type> type, void* value, std::span<Ref> args) {
			auto ty = new std::shared_ptr<Type>[1]{ type };
			auto args_ptr = new Ref[args.size() + 1];
			auto mem_ty = new std::shared_ptr<Type>[1]{ types.memory(0) };
			args_ptr[0] =
			    first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ mem_ty, 1 }, .constant = { .value = value, .owned = static_cast<bool>(value) } }));
			std::copy(args.begin(), args.end(), args_ptr + 1);
			return first(emplace_op(Node{ .kind = Node::CONSTRUCT, .type = std::span{ ty, 1 }, .construct = { .args = std::span(args_ptr, args.size() + 1) } }));
		}

		/// @brief  Allocate the given type, using `src` to describe the allocation.
		Ref make_allocate(std::shared_ptr<Type> type, Ref src, std::optional<Allocator> allocator = {}) {
			auto ty = new std::shared_ptr<Type>[1]{ type };
			return first(emplace_op(Node{ .kind = Node::ALLOCATE, .type = std::span{ ty, 1 }, .allocate = { .src = src, .allocator = allocator } }));
		}

		Ref make_extract(Ref composite, Ref index) {
			auto stripped = Type::stripped(composite.type());
			assert(stripped->kind == Type::ARRAY_TY);
			auto ty = new std::shared_ptr<Type>[3]{ *stripped->array.T, stripped, stripped };
			return first(emplace_op(Node{
			    .kind = Node::SLICE, .type = std::span{ ty, 3 }, .slice = { .src = composite, .start = index, .count = make_constant<uint64_t>(1), .axis = 0 } }));
		}

		Ref make_extract(Ref composite, uint64_t index) {
			auto ty = new std::shared_ptr<Type>[3];
			auto stripped = Type::stripped(composite.type());
			uint8_t axis = 0;
			if (stripped->kind == Type::ARRAY_TY) {
				ty[0] = *stripped->array.T;
			} else if (stripped->kind == Type::COMPOSITE_TY || stripped->kind == Type::UNION_TY) {
				ty[0] = stripped->composite.types[index];
				axis = Node::NamedAxis::FIELD;
			} else {
				assert(0);
			}
			ty[1] = ty[2] = stripped;
			return first(
			    emplace_op(Node{ .kind = Node::SLICE,
			                     .type = std::span{ ty, 3 },
			                     .slice = { .src = composite, .start = make_constant<uint64_t>(index), .count = make_constant<uint64_t>(1), .axis = axis } }));
		}

		Ref make_slice(Ref src, uint8_t axis, Ref base, Ref count) {
			auto stripped = Type::stripped(src.type());
			auto ty = new std::shared_ptr<Type>[3]{ stripped, stripped, stripped };
			return first(emplace_op(Node{ .kind = Node::SLICE, .type = std::span{ ty, 3 }, .slice = { .src = src, .start = base, .count = count, .axis = axis } }));
		}

		Ref make_slice(std::shared_ptr<Type> type_ex, Ref src, uint8_t axis, Ref base, Ref count) {
			auto ty = new std::shared_ptr<Type>[3]{ Type::stripped(type_ex), Type::stripped(src.type()), Type::stripped(src.type()) };
			return first(emplace_op(Node{ .kind = Node::SLICE, .type = std::span{ ty, 3 }, .slice = { .src = src, .start = base, .count = count, .axis = axis } }));
		}
		Ref make_get_allocation_size(Ref ptr) {
			auto ty = new std::shared_ptr<Type>[1]{ types.u64() };
			return first(emplace_op(Node{ .kind = Node::GET_ALLOCATION_SIZE, .type = std::span{ ty, 1 }, .get_allocation_size = { .ptr = ptr } }));
		}
		Ref make_get_ci(Ref ptr);

		// slice splits a range into two halves
		// converge is essentially an unslice -> it returns back to before the slice was made
		// since a slice source is always a single range, converge produces a single range too
		Ref make_converge(std::shared_ptr<Type> type, std::span<Ref> deps) {
			auto stripped = Type::stripped(type);
			auto ty = new std::shared_ptr<Type>[1]{ stripped };

			auto deps_ptr = new Ref[deps.size()];
			std::copy(deps.begin(), deps.end(), deps_ptr);
			return first(emplace_op(Node{ .kind = Node::CONVERGE, .type = std::span{ ty, 1 }, .converge = { .diverged = std::span{ deps_ptr, deps.size() } } }));
		}

		Ref make_use(Ref src, Access acc) {
			auto ty = new std::shared_ptr<Type>[1]{ src.type() };
			return first(emplace_op(Node{ .kind = Node::USE, .type = std::span{ ty, 1 }, .use = { .src = src, .access = acc } }));
		}

		Ref make_cast(std::shared_ptr<Type> dst_type, Ref src) {
			auto ty = new std::shared_ptr<Type>[1]{ dst_type };
			return first(emplace_op(Node{ .kind = Node::CAST, .type = std::span{ ty, 1 }, .cast = { .src = src } }));
		}

		Ref make_acquire_next_image(Ref swapchain) {
			return first(emplace_op(Node{ .kind = Node::ACQUIRE_NEXT_IMAGE,
			                              .type = std::span{ new std::shared_ptr<Type>[1]{ types.make_imageview_ty() }, 1 },
			                              .acquire_next_image = { .swapchain = swapchain } }));
		}

		Ref make_clear_image(Ref dst, Clear cv) {
			return first(emplace_op(Node{ .kind = Node::CLEAR,
			                              .type = std::span{ new std::shared_ptr<Type>[1]{ types.make_imageview_ty() }, 1 },
			                              .clear = { .dst = dst, .cv = new Clear(cv) } }));
		}

		Ref make_declare_fn(std::shared_ptr<Type> const fn_ty) {
			auto ty = new std::shared_ptr<Type>[1]{ fn_ty };
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = nullptr } }));
		}

		template<class... Refs>
		Node* make_call(Ref fn, Refs... args) {
			Ref* args_ptr = new Ref[sizeof...(args) + 1]{ fn, args... };
			decltype(Node::call) call = { .args = std::span(args_ptr, sizeof...(args) + 1) };
			Node n{};
			n.kind = Node::CALL;
			if (fn.type()->kind == Type::OPAQUE_FN_TY) {
				n.type = { new std::shared_ptr<Type>[fn.type()->opaque_fn.return_types.size()], fn.type()->opaque_fn.return_types.size() };
				std::copy(fn.type()->opaque_fn.return_types.begin(), fn.type()->opaque_fn.return_types.end(), n.type.data());
			} else if (fn.type()->kind == Type::SHADER_FN_TY) {
				n.type = { new std::shared_ptr<Type>[fn.type()->shader_fn.return_types.size()], fn.type()->shader_fn.return_types.size() };
				std::copy(fn.type()->shader_fn.return_types.begin(), fn.type()->shader_fn.return_types.end(), n.type.data());
			} else if (fn.type()->kind == Type::MEMORY_TY) { // TODO: typing
				n.type = { new std::shared_ptr<Type>[sizeof...(args)], sizeof...(args) };
				std::fill(n.type.begin(), n.type.end(), types.make_void_ty());
			} else {
				assert(0);
			}

			n.call = call;
			return emplace_op(n);
		}

		Ref make_release(Ref src, Access dst_access = Access::eNone, DomainFlagBits dst_domain = DomainFlagBits::eAny) {
			Ref* args_ptr = new Ref[1]{ src };
			auto tys = new std::shared_ptr<Type>[1]{ Type::stripped(src.type()) };
			return first(emplace_op(Node{ .kind = Node::RELEASE,
			                              .type = std::span{ tys, 1 },
			                              .release = { .src = std::span{ args_ptr, 1 }, .dst_access = dst_access, .dst_domain = dst_domain } }));
		}
		template<class T>
		Ref acquire(std::shared_ptr<Type> type, AcquireRelease* acq_rel, T value) {
			auto val_ptr = new (new std::byte[sizeof(T)]) T(value);

			auto tys = new std::shared_ptr<Type>[1]{ type };
			auto vals = new void*[1]{ val_ptr };

			// spelling this out due to clang bug
			Node node{};
			node.kind = Node::ACQUIRE;
			node.type = std::span{ tys, 1 };
			node.rel_acq = acq_rel;
			node.acquire = {};
			node.acquire.values = std::span{ vals, 1 };
			return first(emplace_op(std::move(node)));
		}

		Ref make_compile_pipeline(Ref src) {
			auto tys = new std::shared_ptr<Type>[1]{ types.memory(sizeof(PipelineBaseInfo*)) };
			return first(emplace_op(Node{ .kind = Node::COMPILE_PIPELINE, .type = std::span{ tys, 1 }, .compile_pipeline = { .src = src } }));
		}

		// MATH

		Ref make_math_binary_op(Node::BinOp op, Ref a, Ref b) {
			std::shared_ptr<Type>* tys = new std::shared_ptr<Type>[1]{ a.type() };

			return first(emplace_op(Node{ .kind = Node::MATH_BINARY, .type = std::span{ tys, 1 }, .math_binary = { .a = a, .b = b, .op = op } }));
		}

		// EDITS

		// GC
		void collect_garbage();
		void collect_garbage(std::pmr::polymorphic_allocator<std::byte> allocator);
	};

	template<class T>
	T* get_value(Ref parm) {
		assert(parm.node->kind == Node::ACQUIRE || parm.node->kind == Node::CONSTANT);
		if (parm.node->kind == Node::ACQUIRE) {
			return reinterpret_cast<T*>(parm.node->acquire.values[parm.index]);
		} else if (parm.node->kind == Node::CONSTANT) {
			assert(parm.index == 0);
			return reinterpret_cast<T*>(parm.node->constant.value);
		}
		assert(false);
		return nullptr;
	};

	inline void* get_value(Ref parm) {
		assert(parm.node->kind == Node::ACQUIRE || parm.node->kind == Node::CONSTANT);
		switch (parm.node->kind) {
		case Node::CONSTANT:
			return parm.node->constant.value;
		case Node::ACQUIRE:
			return parm.node->acquire.values[parm.index];
		default:
			assert(0);
			return nullptr;
		}
	}

	inline std::span<void*> get_values(Node* node) {
		assert(node->kind == Node::ACQUIRE);
		return node->acquire.values;
	}

	extern thread_local std::shared_ptr<IRModule> current_module;

	struct ExtNode {
		ExtNode(Node* node) : node(node) {
			acqrel = new AcquireRelease;
			node->rel_acq = acqrel;
			this->node->held = true;
			source_module = current_module;
		}

		ExtNode(Node* node, std::vector<std::shared_ptr<ExtNode>> deps) : node(node), deps(std::move(deps)) {
			acqrel = new AcquireRelease;
			node->rel_acq = acqrel;
			this->node->held = true;
			source_module = current_module;
		}

		ExtNode(Node* node, std::shared_ptr<ExtNode> dep) : node(node) {
			acqrel = new AcquireRelease;
			node->rel_acq = acqrel;
			this->node->held = true;
			deps.push_back(std::move(dep));

			source_module = current_module;
		}

		ExtNode(Ref ref, std::shared_ptr<ExtNode> dep, Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eAny) {
			acqrel = new AcquireRelease;
			this->node = current_module->make_release(ref, access, domain).node;
			node->rel_acq = acqrel;
			this->node->held = true;
			deps.push_back(std::move(dep));

			source_module = current_module;
		}

		// for acquires - adopt the node
		ExtNode(Node* node, ResourceUse use) : node(node) {
			acqrel = new AcquireRelease;
			acqrel->status = Signal::Status::eHostAvailable;
			acqrel->last_use.resize(1);
			acqrel->last_use[0] = use;

			node->rel_acq = acqrel;
			this->node->held = true;
			source_module = current_module;
		}

		~ExtNode() {
			if (acqrel) {
				node->held = false;
			}
		}

		ExtNode(ExtNode&& o) = delete;
		ExtNode& operator=(ExtNode&& o) = delete;

		Node* get_node() {
			return node;
		}

		void mutate(Node* new_node) {
			node->held = false;
			node = new_node;
			new_node->held = true;
		}

		AcquireRelease* acqrel;
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

	struct ScheduledItem {
		Node* execable;
		DomainFlagBits scheduled_domain;
		Stream* scheduled_stream;
		size_t naming_index;
	};

	struct ExecutionInfo {
		Stream* stream;
		size_t naming_index;
		Node::Kind kind;
	};

	std::string exec_to_string(ScheduledItem& item);
} // namespace vuk
