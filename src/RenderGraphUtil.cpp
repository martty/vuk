#include "vuk/RenderGraph.hpp"
#include "RenderGraphImpl.hpp"
#include "RenderGraphUtil.hpp"

namespace vuk {
#define INIT2(x) x(decltype(x)::allocator_type(arena_))

	RenderPassInfo::RenderPassInfo(arena& arena_) : INIT2(subpasses), INIT2(attachments) {
	}

    PassInfo::PassInfo(arena& arena_, Pass&& p) : pass(std::move(p)) {}

    SubpassInfo::SubpassInfo(arena& arena_) : INIT2(passes) {}

#undef INIT2

	// implement MapProxy for relevant types

	// implement MapProxy for UseRefs
	using MP1 = MapProxy<Name, std::span<const UseRef>>;
	using MPI1 = ConstMapIterator<Name, std::span<const UseRef>>;
	using M1 = robin_hood::unordered_flat_map<Name, std::vector<UseRef, short_alloc<UseRef, 64>>>;
	MP1::const_iterator MP1::cbegin() const noexcept {
		auto& map = *reinterpret_cast<M1*>(_map);
		return MP1::const_iterator(new M1::const_iterator(map.cbegin()));
	}

	MP1::const_iterator MP1::cend() const noexcept {
		auto& map = *reinterpret_cast<M1*>(_map);
		return MP1::const_iterator(new M1::const_iterator(map.cend()));
	}

	MP1::const_iterator MP1::find(Name key) const noexcept {
		auto& map = *reinterpret_cast<M1*>(_map);
		return MP1::const_iterator(new M1::const_iterator(map.find(key)));
	}

	size_t MP1::size() const noexcept {
		auto& map = *reinterpret_cast<M1*>(_map);
		return map.size();
	}

	template<>
	MPI1::~ConstMapIterator() {
		delete reinterpret_cast<M1::const_iterator*>(_iter);
	}

	MPI1::ConstMapIterator(const MPI1& other) noexcept {
		*reinterpret_cast<M1::const_iterator*>(_iter) = *reinterpret_cast<M1::iterator*>(other._iter);
	}

	MPI1::reference MPI1::operator*() noexcept {
		const auto& iter = *reinterpret_cast<M1::const_iterator const*>(_iter);
		std::pair<const Name&, std::span<const UseRef>> result(iter->first, std::span(iter->second));
		return result;
	}

	MPI1& MPI1::operator++() noexcept {
		reinterpret_cast<M1::iterator*>(_iter)->operator++();
		return *this;
	}

	bool MPI1::operator==(MPI1 const& other) const noexcept {
		return *reinterpret_cast<M1::iterator const*>(_iter) == *reinterpret_cast<M1::iterator const*>(other._iter);
	}
	

	// implement MapProxy for attachment
	using MP2 = MapProxy<Name, AttachmentRPInfo&>;
	using MPI2 = ConstMapIterator<Name, AttachmentRPInfo&>;
	using M2 = robin_hood::unordered_flat_map<Name, AttachmentRPInfo&>;
	MP2::const_iterator MP2::cbegin() const noexcept {
		auto& map = *reinterpret_cast<M2*>(_map);
		return MP2::const_iterator(new M2::const_iterator(map.cbegin()));
	}

	MP2::const_iterator MP2::cend() const noexcept {
		auto& map = *reinterpret_cast<M2*>(_map);
		return MP2::const_iterator(new M2::const_iterator(map.cend()));
	}

	MP2::const_iterator MP2::find(Name key) const noexcept {
		auto& map = *reinterpret_cast<M2*>(_map);
		return MP2::const_iterator(new M2::const_iterator(map.find(key)));
	}

	size_t MP2::size() const noexcept {
		auto& map = *reinterpret_cast<M2*>(_map);
		return map.size();
	}

	template<>
	MPI2::~ConstMapIterator() {
		delete reinterpret_cast<M2::const_iterator*>(_iter);
	}

	MPI2::ConstMapIterator(const MPI2& other) noexcept {
		*reinterpret_cast<M2::const_iterator*>(_iter) = *reinterpret_cast<M2::iterator*>(other._iter);
	}

	MPI2::reference MPI2::operator*() noexcept {
		const auto& iter = *reinterpret_cast<M2::const_iterator const*>(_iter);
		std::pair<const Name&, AttachmentRPInfo&> result(iter->first, iter->second);
		return result;
	}

	MPI2& MPI2::operator++() noexcept {
		reinterpret_cast<M2::iterator*>(_iter)->operator++();
		return *this;
	}

	bool MPI2::operator==(MPI2 const& other) const noexcept {
		return *reinterpret_cast<M2::iterator const*>(_iter) == *reinterpret_cast<M2::iterator const*>(other._iter);
	}
} // namespace vuk
