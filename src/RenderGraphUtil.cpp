#include "RenderGraphUtil.hpp"
#include "RenderGraphImpl.hpp"
#include "vuk/RenderGraph.hpp"

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

	RenderPassInfo::RenderPassInfo(arena& arena_) : INIT2(subpasses), INIT2(attachments) {}

	PassInfo::PassInfo(arena& arena_, Pass& p) :
	    pass(&p),
	    INIT2(input_names),
	    INIT2(output_names),
	    INIT2(write_input_names) {}

	SubpassInfo::SubpassInfo(arena& arena_) : INIT2(passes) {}

#undef INIT2

	// implement MapProxy for relevant types

	// implement MapProxy for UseRefs
	using MP1 = MapProxy<Name, std::span<const UseRef>>;
	using MPI1 = ConstMapIterator<Name, std::span<const UseRef>>;
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
	MP1::const_iterator MP1::find(Name key) const noexcept {
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
		std::pair<const Name&, std::span<const UseRef>> result(iter->first, std::span(iter->second));
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
	using MP2 = MapProxy<Name, const AttachmentInfo&>;
	using MPI2 = ConstMapIterator<Name, const AttachmentInfo&>;
	using M2 = robin_hood::unordered_flat_map<Name, AttachmentInfo>;

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
	MP2::const_iterator MP2::find(Name key) const noexcept {
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
	using MP3 = MapProxy<Name, const BufferInfo&>;
	using MPI3 = ConstMapIterator<Name, const BufferInfo&>;
	using M3 = robin_hood::unordered_flat_map<Name, BufferInfo>;

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
	MP3::const_iterator MP3::find(Name key) const noexcept {
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
} // namespace vuk
