#pragma once

#include "vuk/Allocator.hpp"
#include "vuk/Context.hpp"
#include "vuk/IR.hpp"
#include "vuk/ImageAttachment.hpp"
#include "vuk/Types.hpp"
#include "vuk/vuk_fwd.hpp"

#include <memory>
#include <span>
#include <variant>

// futures
namespace vuk {
	class FutureControlBlock {
	public:
		/// @brief Submit Future for execution
		Result<void> submit(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});
		/// @brief Submit and wait for Future to complete execution on host
		Result<void> wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {});

		/// @brief If the Future has been submitted for execution, polls for status.
		[[nodiscard]] Result<Signal::Status> poll();

		AcquireRelease acqrel;

		std::shared_ptr<RG> rg;

		/// @brief Get the referenced RenderGraph
		std::shared_ptr<RG>& get_render_graph() {
			return rg;
		}
	};

	template<class T>
	class Future;

	class UntypedFuture {
	public:
		UntypedFuture(std::shared_ptr<RG> rg, Ref ref, Ref def) {
			this->control = std::make_shared<FutureControlBlock>();

			this->head = { rg->make_release(ref, &this->control->acqrel, Access::eHostRW, DomainFlagBits::eHost), 0 };

			this->control->rg = std::move(rg);
			this->def = def;
		}

		UntypedFuture(const UntypedFuture& o) noexcept : control{ std::make_shared<FutureControlBlock>(*o.control) }, def{ o.def } {
			head = { control->rg->make_release(o.get_head(), &this->control->acqrel, Access::eHostRW, DomainFlagBits::eHost), 0 };
		}

		UntypedFuture(UntypedFuture&& o) noexcept :
		    control{ std::exchange(o.control, nullptr) },
		    def{ std::exchange(o.def, {}) },
		    head{ std::exchange(o.head, {}) } {}

		UntypedFuture& operator=(const UntypedFuture& o) noexcept {
			control = { std::make_shared<FutureControlBlock>(*o.control) };
			def = { o.def };

			head = { control->rg->make_release(o.get_head(), &this->control->acqrel, Access::eHostRW, DomainFlagBits::eHost), 0 };

			return *this;
		}

		UntypedFuture& operator=(UntypedFuture&& o) noexcept {
			std::swap(o.control, control);
			std::swap(o.def, def);
			std::swap(o.head, head);

			return *this;
		}

		~UntypedFuture() {
			abandon();
		}

		/// @brief Get the referenced RenderGraph
		const std::shared_ptr<RG>& get_render_graph() const noexcept {
			return control->rg;
		}

		/// @brief Name the value currently referenced by this Future
		void set_name(std::string_view name) noexcept {
			get_render_graph()->name_output(head, std::string(name));
		}

		Ref get_head() const noexcept {
			return head.node->release.src;
		}

		Ref get_def() const noexcept {
			return def;
		}

		Result<void> wait(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {}) {
			return control->wait(allocator, compiler, options);
		}

		template<class U>
		Future<U> transmute(Ref ref) noexcept {
			head.node->release.src = ref;
			return *reinterpret_cast<Future<U>*>(this); // TODO: not cool
		}

		void abandon() {
			if (head.node) {
				assert(head.node->kind == Node::RELEASE || head.node->kind == Node::NOP);
				head.node->kind = Node::NOP;
			}
		}

		// TODO: remove this from public API
		std::shared_ptr<FutureControlBlock> control;

	protected:
		Ref def;
		Ref head;
	};

	template<class T>
	class Future : public UntypedFuture {
	public:
		using UntypedFuture::UntypedFuture;

		template<class U = T>
		Future<U> transmute(Ref ref) noexcept {
			head.node->release.src = ref;
			return *reinterpret_cast<Future<U>*>(this); // TODO: not cool
		}

		template<class U = T>
		Future<U> release_to(Ref ref, Access access, DomainFlagBits domain) noexcept {
			head.node->release.src = ref;
			head.node->release.dst_access = access;
			head.node->release.dst_domain = domain;
			return std::move(*reinterpret_cast<Future<U>*>(this)); // TODO: not cool
		}

		T* operator->() noexcept {
			return reinterpret_cast<T*>(def.node->valloc.args[0].node->constant.value);
		}

		/// @brief Wait and retrieve the result of the Future on the host
		[[nodiscard]] Result<T> get(Allocator& allocator, Compiler& compiler, RenderGraphCompileOptions options = {})
		  requires(!std::is_array_v<T>)
		{
			if (auto result = control->wait(allocator, compiler, options)) {
				return { expected_value, *operator->() };
			} else {
				return result;
			}
		}

		// Image inferences
		void same_extent_as(const Future<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			assert(src.get_def().type()->is_image());
			def.node->valloc.args[1] = src.get_def().node->valloc.args[1];
			def.node->valloc.args[2] = src.get_def().node->valloc.args[2];
			def.node->valloc.args[3] = src.get_def().node->valloc.args[3];
		}

		/// @brief Inference target has the same width & height as the source
		void same_2D_extent_as(const Future<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			assert(src.get_def().type()->is_image());
			def.node->valloc.args[1] = src.get_def().node->valloc.args[1];
			def.node->valloc.args[2] = src.get_def().node->valloc.args[2];
		}

		/// @brief Inference target has the same format as the source
		void same_format_as(const Future<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			assert(src.get_def().type()->is_image());
			def.node->valloc.args[4] = src.get_def().node->valloc.args[4];
		}

		/// @brief Inference target has the same shape(extent, layers, levels) as the source
		void same_shape_as(const Future<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			same_extent_as(src);
			for (auto i = 6; i < 10; i++) { /* 6 - 9 : layers, levels */
				def.node->valloc.args[i] = src.get_def().node->valloc.args[i];
			}
		}

		/// @brief Inference target is similar to(same shape, same format, same sample count) the source
		void similar_to(const Future<ImageAttachment>& src)
		  requires std::is_same_v<T, ImageAttachment>
		{
			same_shape_as(src);
			same_format_as(src);
			def.node->valloc.args[5] = src.get_def().node->valloc.args[5]; // sample count
		}

		// Buffer inferences

		void same_size(const Future<Buffer>& src)
		  requires std::is_same_v<T, Buffer>
		{
			assert(src.get_def().type()->is_buffer());
			def.node->valloc.args[1] = src.get_def().node->valloc.args[1];
		}

		Future<uint64_t> get_size()
		  requires std::is_same_v<T, Buffer>
		{
			return { get_render_graph(), def.node->valloc.args[1], {} };
		}

		void set_size(Future<uint64_t> arg)
		  requires std::is_same_v<T, Buffer>
		{
			get_render_graph()->subgraphs.push_back(arg.get_render_graph());
			def.node->valloc.args[1] = arg.get_head();
		}

		auto operator[](size_t index)
		  requires std::is_array_v<T>
		{
			auto item_def = def.node->aalloc.defs[index];
			Ref item = control->rg->make_array_indexing(def.type()->array.T, get_head(), control->rg->make_constant(index));
			assert(def.node->kind == Node::AALLOC);
			assert(def.type()->kind == Type::ARRAY_TY);
			return Future<std::remove_reference_t<decltype(std::declval<T>()[0])>>(get_render_graph(), item, item_def);
		}
	};

	inline Future<uint64_t> operator*(Future<uint64_t> a, uint64_t b) {
		Ref ref = a.get_render_graph()->make_math_binary_op(Node::BinOp::MUL, a.get_head(), a.get_render_graph()->make_constant(b));
		return std::move(std::move(a).transmute<uint64_t>(ref));
	}

	inline Result<void> wait_for_futures_explicit(Allocator& alloc, Compiler& compiler, std::span<UntypedFuture> futures) {
		std::vector<std::shared_ptr<RG>> rgs_to_run;
		for (uint64_t i = 0; i < futures.size(); i++) {
			auto& future = futures[i];
			if (future.control->acqrel.status == Signal::Status::eDisarmed && !futures[i].get_render_graph()) {
				return { expected_error, RenderGraphException{} };
			} else if (future.control->acqrel.status == Signal::Status::eHostAvailable || future.control->acqrel.status == Signal::Status::eSynchronizable) {
				continue;
			} else {
				rgs_to_run.emplace_back(futures[i].get_render_graph());
			}
		}
		if (rgs_to_run.size() != 0) {
			VUK_DO_OR_RETURN(link_execute_submit(alloc, compiler, std::span(rgs_to_run)));
		}

		std::vector<SyncPoint> waits;
		for (uint64_t i = 0; i < futures.size(); i++) {
			auto& future = futures[i];
			if (future.control->acqrel.status != Signal::Status::eSynchronizable) {
				continue;
			}
			waits.emplace_back(future.control->acqrel.source);
		}
		if (waits.size() > 0) {
			alloc.get_context().wait_for_domains(std::span(waits));
		}

		return { expected_value };
	}

	template<class... Args>
	Result<void> wait_for_futures(Allocator& alloc, Compiler& compiler, Args&&... futs) {
		return wait_for_futures_explicit(alloc, compiler, std::array{ futs... });
	}
} // namespace vuk