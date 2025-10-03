#pragma once

#include "vuk/Buffer.hpp"
#include "vuk/Exception.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/RelSpan.hpp"
#include "vuk/ResourceUse.hpp"
#include "vuk/Result.hpp"
#include "vuk/runtime/vk/Allocator.hpp"

#include <atomic>
#include <function2/function2.hpp>
#include <optional>
#include <plf_colony.h>
#include <span>
#include <vector>

// #define VUK_GARBAGE_SAN

namespace vuk {
	struct IRModule;

	struct TypeDebugInfo {
		std::string name;
	};

	using UserCallbackType = fu2::unique_function<void(CommandBuffer&, std::span<void*>, std::span<void*>, std::span<void*>)>;

	struct Type {
		enum struct TypeKind { VOID_TY = 0, MEMORY_TY = 1, INTEGER_TY, COMPOSITE_TY, ARRAY_TY, UNION_TY, IMBUED_TY, ALIASED_TY, OPAQUE_FN_TY, SHADER_FN_TY } kind;
		using enum TypeKind;

		using Hash = uint32_t;

		size_t size = ~0ULL;
		Hash hash_value;

		TypeDebugInfo debug_info;

		std::vector<std::shared_ptr<Type>> child_types;
		std::vector<size_t> offsets;                // for now only useful for composites
		std::unique_ptr<UserCallbackType> callback; // only useful for user CBs

		union {
			struct {
				uint32_t width;
			} integer;
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
				int execute_on;
			} opaque_fn;
			struct {
				void* shader;
				std::span<std::shared_ptr<Type>> args;
				std::span<std::shared_ptr<Type>> return_types;
				int execute_on;
			} shader_fn;
			struct {
				std::shared_ptr<Type>* T;
				size_t count;
				size_t stride;
			} array;
			struct {
				std::span<std::shared_ptr<Type>> types;
				size_t tag;
			} composite;
		};

		~Type() = default;

		[[nodiscard]] static std::shared_ptr<Type> stripped(std::shared_ptr<Type> t);

		[[nodiscard]] static std::shared_ptr<Type> extract(std::shared_ptr<Type> t, size_t index);

		[[nodiscard]] static Hash hash_integer(size_t width);
		[[nodiscard]] static Hash hash(Type const* t);

		[[nodiscard]] static std::string_view to_sv(Access acc);
		[[nodiscard]] static std::string to_string(Type* t);
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
			SLICE,
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
		AcquireRelease* rel_acq = nullptr;
		bool held = false;

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
				std::span<Ref> defs; // for preserving provenance for composite types
				std::optional<Allocator> allocator;
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
				Access access;
			} use;
			struct : Fixed<1> {
				Ref src;
			} logical_copy;
			struct : Fixed<1> {
				Ref dst;
				Ref value;
				size_t index;
			} set;
			struct : Fixed<1> {
				Ref src;
			} compile_pipeline;
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

		[[nodiscard]] static std::string_view kind_to_sv(Node::Kind kind);
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

	struct CannotBeConstantEvaluated : Exception {
		CannotBeConstantEvaluated(Ref ref) : ref(ref) {}

		Ref ref;

		void throw_this() override {
			throw *this;
		}
	};

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

	Result<RefOrValue, CannotBeConstantEvaluated> eval(Ref ref);

	template<class T>
	  requires(!std::is_pointer_v<T>)
	Result<T, CannotBeConstantEvaluated> eval(Ref ref) {
		auto res = eval(ref);
		if (!res) {
			return res;
		}
		if (res.holds_value() && !res->is_ref) {
			return { expected_value, *static_cast<T*>(res->value) };
		} else {
			return { expected_control, CannotBeConstantEvaluated{ ref } };
		}
	}

	template<class T>
	T& constant(Ref ref) {
		assert(ref.type()->kind == Type::INTEGER_TY || ref.type()->kind == Type::MEMORY_TY);
		return *reinterpret_cast<T*>(ref.node->constant.value);
	}

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

		void deallocate([[maybe_unused]] T* p, [[maybe_unused]] std::size_t n) noexcept {}

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
			types.get_builtin_buffer();
			types.get_builtin_image();
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

			Type::Hash builtin_image = 0;
			Type::Hash builtin_buffer = 0;
			Type::Hash builtin_swapchain = 0;
			Type::Hash builtin_sampler = 0;
			Type::Hash builtin_sampled_image = 0;

			size_t union_tag_type_counter = 0;

			// TYPES
			std::shared_ptr<Type> make_void_ty();
			std::shared_ptr<Type> make_imbued_ty(std::shared_ptr<Type> ty, Access access);
			std::shared_ptr<Type> make_aliased_ty(std::shared_ptr<Type> ty, size_t ref_idx);
			std::shared_ptr<Type> make_array_ty(std::shared_ptr<Type> ty, size_t count);
			std::shared_ptr<Type> make_union_ty(std::vector<std::shared_ptr<Type>> types);
			std::shared_ptr<Type> make_opaque_fn_ty(std::span<std::shared_ptr<Type> const> args,
			                                        std::span<std::shared_ptr<Type> const> ret_types,
			                                        DomainFlags execute_on,
			                                        size_t hash_code,
			                                        UserCallbackType callback,
			                                        std::string_view name);
			std::shared_ptr<Type> make_shader_fn_ty(std::span<std::shared_ptr<Type> const> args,
			                                        std::span<std::shared_ptr<Type> const> ret_types,
			                                        DomainFlags execute_on,
			                                        void* shader,
			                                        std::string_view name);
			std::shared_ptr<Type> u64();
			std::shared_ptr<Type> u32();
			std::shared_ptr<Type> memory(size_t size);
			std::shared_ptr<Type> get_builtin_image();
			std::shared_ptr<Type> get_builtin_buffer();
			std::shared_ptr<Type> get_builtin_swapchain();
			std::shared_ptr<Type> get_builtin_sampler();
			std::shared_ptr<Type> get_builtin_sampled_image();
			std::shared_ptr<Type> emplace_type(std::shared_ptr<Type> t);
			TypeDebugInfo allocate_type_debug_info(std::string name);
			void collect();
			void destroy(Type* t, void* v);
		} types;

		Node* emplace_op(Node v);
		void name_output(Ref ref, std::string_view name);
		void set_source_location(Node* node, SourceLocationAtFrame loc);
		std::optional<plf::colony<Node>::iterator> destroy_node(Node* node);
		// OPS

		Ref make_constant(std::shared_ptr<Type> type, void* value);

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
			return first(
			    emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = new (new char[sizeof(T)]) T(value), .owned = true } }));
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
			return first(emplace_op(Node{ .kind = Node::CONSTANT, .type = std::span{ ty, 1 }, .constant = { .value = value, .owned = false } }));
		}

		void set_value(Ref ref, size_t index, Ref value);

		template<class T>
		void set_value(Ref ref, size_t index, T value) {
			auto co = make_constant(value);
			emplace_op(Node{ .kind = Node::SET, .set = { .dst = ref, .value = co, .index = index } });
		}

		Ref make_declare_image(ImageAttachment value);

		Ref make_declare_buffer(Buffer value);
		Ref make_declare_array(std::shared_ptr<Type> type, std::span<Ref> args);
		Ref make_declare_union(std::span<Ref> args);
		Ref make_declare_swapchain(Swapchain& bundle);
		Ref make_sampled_image(Ref image, Ref sampler);
		Ref make_extract(Ref composite, Ref index);
		Ref make_extract(Ref composite, uint64_t index);
		Ref make_slice(Ref src, uint8_t axis, Ref base, Ref count);
		Ref make_slice(std::shared_ptr<Type> type_ex, Ref src, uint8_t axis, Ref base, Ref count);
		// slice splits a range into two halves
		// converge is essentially an unslice -> it returns back to before the slice was made
		// since a slice source is always a single range, converge produces a single range too
		Ref make_converge(std::shared_ptr<Type> type, std::span<Ref> deps);
		Ref make_use(Ref src, Access acc);
		Ref make_cast(std::shared_ptr<Type> dst_type, Ref src);
		Ref make_acquire_next_image(Ref swapchain);
		Ref make_clear_image(Ref dst, Clear cv);
		Ref make_declare_fn(std::shared_ptr<Type> const fn_ty);

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

		Ref make_release(Ref src, Access dst_access = Access::eNone, DomainFlagBits dst_domain = DomainFlagBits::eAny);

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

		Ref make_compile_pipeline(Ref src);

		// MATH

		Ref make_math_binary_op(Node::BinOp op, Ref a, Ref b);

		// GC

		void collect_garbage();
		void collect_garbage(std::pmr::polymorphic_allocator<std::byte> allocator);
	};

	extern thread_local std::shared_ptr<IRModule> current_module;

	struct ExtNode {
		ExtNode(Node* node);
		ExtNode(Node* node, std::vector<std::shared_ptr<ExtNode>> deps);
		ExtNode(Node* node, std::shared_ptr<ExtNode> dep);
		ExtNode(Ref ref, std::shared_ptr<ExtNode> dep, Access access = Access::eNone, DomainFlagBits domain = DomainFlagBits::eAny);
		// for acquires - adopt the node
		ExtNode(Node* node, ResourceUse use);
		~ExtNode();
		ExtNode(ExtNode&& o) = delete;
		ExtNode& operator=(ExtNode&& o) = delete;

		[[nodiscard]] Node* get_node() {
			return node;
		}

		void mutate(Node* new_node);

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

	[[nodiscard]] std::string exec_to_string(ScheduledItem& item);
} // namespace vuk
