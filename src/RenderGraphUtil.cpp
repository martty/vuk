#include "RenderGraphUtil.hpp"
#include "RenderGraphImpl.hpp"
#include "vuk/RenderGraph.hpp"
#include <fmt/format.h>

namespace vuk {
	namespace detail {
		ImageResourceInputOnly ImageResource::operator>>(Access ia) {
			return { name, ia };
		}

		Resource ImageResourceInputOnly::operator>>(Name out) {
			return { name, Resource::Type::eImage, ba, out };
		}

		ImageResourceInputOnly::operator Resource() {
			if (!is_write_access(ba)) { // do not produce a name by default it is read-only
				return operator>>(Name{});
			}
			return operator>>(name.append("+"));
		}

		BufferResourceInputOnly BufferResource::operator>>(Access ba) {
			return { name, ba };
		}

		Resource BufferResourceInputOnly::operator>>(Name out) {
			return { name, Resource::Type::eBuffer, ba, out };
		}

		BufferResourceInputOnly::operator Resource() {
			if (!is_write_access(ba)) { // do not produce a name by default it is read-only
				return operator>>(Name{});
			}
			return operator>>(name.append("+"));
		}
	} // namespace detail

#define INIT2(x) x(decltype(x)::allocator_type(arena_))

	RenderPassInfo::RenderPassInfo(arena& arena_) :
	    INIT2(subpasses),
	    INIT2(attachments),
	    INIT2(pre_barriers),
	    INIT2(post_barriers),
	    INIT2(pre_mem_barriers),
	    INIT2(post_mem_barriers),
	    INIT2(waits) {}

	SubpassInfo::SubpassInfo(arena& arena_) : INIT2(passes), INIT2(pre_barriers), INIT2(post_barriers), INIT2(pre_mem_barriers), INIT2(post_mem_barriers) {}

#undef INIT2

	// implement MapProxy for relevant types

	// implement MapProxy for UseRefs
	using MP1 = MapProxy<QualifiedName, std::span<const UseRef>>;
	using MPI1 = ConstMapIterator<QualifiedName, std::span<const UseRef>>;
	using M1 = decltype(RGCImpl::use_chains);

	template<>
	MP1::const_iterator MP1::cbegin() const noexcept {
		auto& map = *reinterpret_cast<M1*>(_map);
		return MP1::const_iterator(new M1::const_iterator(map.cbegin()));
	}

	template<>
	MP1::const_iterator MP1::cend() const noexcept {
		auto& map = *reinterpret_cast<M1*>(_map);
		return MP1::const_iterator(new M1::const_iterator(map.cend()));
	}

	template<>
	MP1::const_iterator MP1::find(QualifiedName key) const noexcept {
		auto& map = *reinterpret_cast<M1*>(_map);
		return MP1::const_iterator(new M1::const_iterator(map.find(key)));
	}

	template<>
	size_t MP1::size() const noexcept {
		auto& map = *reinterpret_cast<M1*>(_map);
		return map.size();
	}

	template<>
	MPI1::~ConstMapIterator() {
		delete reinterpret_cast<M1::const_iterator*>(_iter);
	}

	template<>
	MPI1::ConstMapIterator(const MPI1& other) noexcept {
		*reinterpret_cast<M1::const_iterator*>(_iter) = *reinterpret_cast<M1::iterator*>(other._iter);
	}

	template<>
	MPI1::reference MPI1::operator*() noexcept {
		const auto& iter = *reinterpret_cast<M1::const_iterator const*>(_iter);
		std::pair<const QualifiedName&, std::span<const UseRef>> result(iter->first, std::span(iter->second));
		return result;
	}

	template<>
	MPI1& MPI1::operator++() noexcept {
		reinterpret_cast<M1::iterator*>(_iter)->operator++();
		return *this;
	}

	template<>
	bool MPI1::operator==(MPI1 const& other) const noexcept {
		return *reinterpret_cast<M1::iterator const*>(_iter) == *reinterpret_cast<M1::iterator const*>(other._iter);
	}

	// implement MapProxy for attachment
	using MP2 = MapProxy<QualifiedName, const AttachmentInfo&>;
	using MPI2 = ConstMapIterator<QualifiedName, const AttachmentInfo&>;
	using M2 = robin_hood::unordered_flat_map<QualifiedName, AttachmentInfo>;

	template<>
	MP2::const_iterator MP2::cbegin() const noexcept {
		auto& map = *reinterpret_cast<M2*>(_map);
		return MP2::const_iterator(new M2::const_iterator(map.cbegin()));
	}

	template<>
	MP2::const_iterator MP2::cend() const noexcept {
		auto& map = *reinterpret_cast<M2*>(_map);
		return MP2::const_iterator(new M2::const_iterator(map.cend()));
	}

	template<>
	MP2::const_iterator MP2::find(QualifiedName key) const noexcept {
		auto& map = *reinterpret_cast<M2*>(_map);
		return MP2::const_iterator(new M2::const_iterator(map.find(key)));
	}

	template<>
	size_t MP2::size() const noexcept {
		auto& map = *reinterpret_cast<M2*>(_map);
		return map.size();
	}

	template<>
	MPI2::~ConstMapIterator() {
		delete reinterpret_cast<M2::const_iterator*>(_iter);
	}

	template<>
	MPI2::ConstMapIterator(const MPI2& other) noexcept {
		*reinterpret_cast<M2::const_iterator*>(_iter) = *reinterpret_cast<M2::iterator*>(other._iter);
	}

	template<>
	MPI2::reference MPI2::operator*() noexcept {
		const auto& iter = *reinterpret_cast<M2::const_iterator const*>(_iter);
		return { iter->first, iter->second };
	}

	template<>
	MPI2& MPI2::operator++() noexcept {
		reinterpret_cast<M2::iterator*>(_iter)->operator++();
		return *this;
	}

	template<>
	bool MPI2::operator==(MPI2 const& other) const noexcept {
		return *reinterpret_cast<M2::iterator const*>(_iter) == *reinterpret_cast<M2::iterator const*>(other._iter);
	}

	// implement MapProxy for attachment
	using MP3 = MapProxy<QualifiedName, const BufferInfo&>;
	using MPI3 = ConstMapIterator<QualifiedName, const BufferInfo&>;
	using M3 = robin_hood::unordered_flat_map<QualifiedName, BufferInfo>;

	template<>
	MP3::const_iterator MP3::cbegin() const noexcept {
		auto& map = *reinterpret_cast<M3*>(_map);
		return MP3::const_iterator(new M3::const_iterator(map.cbegin()));
	}

	template<>
	MP3::const_iterator MP3::cend() const noexcept {
		auto& map = *reinterpret_cast<M3*>(_map);
		return MP3::const_iterator(new M3::const_iterator(map.cend()));
	}

	template<>
	MP3::const_iterator MP3::find(QualifiedName key) const noexcept {
		auto& map = *reinterpret_cast<M3*>(_map);
		return MP3::const_iterator(new M3::const_iterator(map.find(key)));
	}

	template<>
	size_t MP3::size() const noexcept {
		auto& map = *reinterpret_cast<M3*>(_map);
		return map.size();
	}

	template<>
	MPI3::~ConstMapIterator() {
		delete reinterpret_cast<M3::const_iterator*>(_iter);
	}

	template<>
	MPI3::ConstMapIterator(const MPI3& other) noexcept {
		*reinterpret_cast<M3::const_iterator*>(_iter) = *reinterpret_cast<M3::iterator*>(other._iter);
	}

	template<>
	MPI3::reference MPI3::operator*() noexcept {
		const auto& iter = *reinterpret_cast<M3::const_iterator const*>(_iter);
		return { iter->first, iter->second };
	}

	template<>
	MPI3& MPI3::operator++() noexcept {
		reinterpret_cast<M3::iterator*>(_iter)->operator++();
		return *this;
	}

	template<>
	bool MPI3::operator==(MPI3 const& other) const noexcept {
		return *reinterpret_cast<M3::iterator const*>(_iter) == *reinterpret_cast<M3::iterator const*>(other._iter);
	}

	namespace errors {
		std::string format_source_location(PassInfo& pass_info) {
			return fmt::format("{}({})", pass_info.pass->source.file_name(), pass_info.pass->source.line());
		}

		RenderGraphException make_unattached_resource_exception(PassInfo& pass_info, Resource& resource, QualifiedName undiverged_name) {
			const char* type = resource.type == Resource::Type::eBuffer ? "buffer" : "image";

			std::string message =
			    fmt::format("{}: Pass <{}> references {} <{}> (also known as <{}::{}>), which was never attached.\n(did you forget an attach_* call?).",
			                format_source_location(pass_info),
			                pass_info.pass->name.c_str(),
			                type,
			                resource.name.name.c_str(),
			                undiverged_name.prefix.c_str(),
			                undiverged_name.name.c_str());
			return RenderGraphException(std::move(message));
		}

		RenderGraphException make_cbuf_references_unknown_resource(PassInfo& pass_info, Resource::Type res_type, Name name) {
			const char* type = res_type == Resource::Type::eBuffer ? "buffer" : "image";
			std::string message = fmt::format("{}: Pass <{}> has attempted to reference {} <{}>, but this name is not known to the rendergraph.",
			                                  format_source_location(pass_info),
			                                  pass_info.pass->name.c_str(),
			                                  type,
			                                  name.c_str());
			return RenderGraphException(std::move(message));
		}

		RenderGraphException make_cbuf_references_undeclared_resource(PassInfo& pass_info, Resource::Type res_type, Name name) {
			const char* type = res_type == Resource::Type::eBuffer ? "buffer" : "image";
			std::string message = fmt::format("{}: In pass <{}>, attempted to bind {} <{}>, but this pass did not declare this name for use.",
			                                  format_source_location(pass_info),
			                                  pass_info.pass->name.c_str(),
			                                  type,
			                                  name.c_str());
			return RenderGraphException(std::move(message));
		}
	} // namespace errors
} // namespace vuk
