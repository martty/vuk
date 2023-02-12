#include "vuk/Name.hpp"
#include "vuk/Hash.hpp"
#include <array>
#include <robin_hood.h>
#include <shared_mutex>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace {
	struct Intern {
		static constexpr size_t arr_siz = 2048;

		const char* add(std::string_view s) {
			auto hash = hash::fnv1a::hash(s.data(), (uint32_t)s.size(), hash::fnv1a::default_offset_basis);
			{
				std::shared_lock _(lock);
				if (auto it = map.find(hash); it != map.end()) {
					return it->second;
				}
			}

			std::unique_lock _(lock);
			// second lookup, under a unique lock, so there are no races
			if (auto it = map.find(hash); it != map.end()) {
				return it->second;
			}

			for (auto& [size, bf] : buffers) {
				auto buffer = std::string_view(bf->data(), bf->size());
				auto pos = buffer.find(s);
				while (pos != std::string::npos && buffer[pos + s.size()] != '\0') {
					pos = buffer.find(s, pos + 1);
				}
				if (pos == std::string_view::npos) {
					if ((size + s.size() + 1) < bf->size()) {
						auto osize = size;
						s.copy(bf->data() + size, s.size());
						*(bf->data() + size + s.size()) = '\0';
						size += s.size() + 1;
						map.emplace(hash, bf->data() + osize);
						return bf->data() + osize;
					}
				} else { // for returning tail substrings
					map.emplace(hash, bf->data() + pos);
					return bf->data() + pos;
				}
			}
			buffers.resize(buffers.size() + 1);
			auto& [nsize, nbuf] = buffers.back();
			nbuf = new std::array<char, arr_siz>{};
			s.copy(nbuf->data(), s.size());
			*(nbuf->data() + s.size()) = '\0';
			nsize += s.size() + 1;
			map.emplace(hash, nbuf->data());
			return nbuf->data();
		}

		Intern() {
			buffers.resize(1);
			buffers[0].first = 1;
			buffers[0].second = new std::array<char, arr_siz>;
			buffers[0].second->at(0) = '\0';
		}

		// to store the strings
		std::vector<std::pair<size_t, std::array<char, arr_siz>*>> buffers;
		robin_hood::unordered_flat_map<uint32_t, const char*> map;
		std::shared_mutex lock;
	};

	static Intern g_intern;
} // namespace

namespace vuk {
	Name::Name(const char* str) noexcept {
		id = g_intern.add(str);
	}

	Name::Name(std::string_view str) noexcept {
		id = g_intern.add(str);
	}

	std::string_view Name::to_sv() const noexcept {
		return id;
	}

	bool Name::is_invalid() const noexcept {
		return id == &invalid_value[0];
	}

	Name Name::append(std::string_view other) const noexcept {
		auto ourlen = strlen(id);
		auto theirlen = other.size();
		auto hash = hash::fnv1a::hash(id, (uint32_t)ourlen, hash::fnv1a::default_offset_basis);
		hash = hash::fnv1a::hash(other.data(), (uint32_t)theirlen, hash);

		// speculative
		{
			std::shared_lock _(g_intern.lock);
			if (auto it = g_intern.map.find(hash); it != g_intern.map.end()) {
				Name n;
				n.id = it->second;
				return n;
			}
		}

		std::string app;
		app.reserve(ourlen + theirlen);
		app.append(id);
		app.append(other);
		return Name(app);
	}
} // namespace vuk

namespace std {
	size_t hash<vuk::Name>::operator()(vuk::Name const& s) const {
		return hash<const char*>()(s.id);
	}

	size_t hash<vuk::QualifiedName>::operator()(vuk::QualifiedName const& s) const {
		uint32_t h = hash<vuk::Name>()(s.prefix);
		::hash_combine_direct(h, hash<vuk::Name>()(s.name));
		return h;
	}
} // namespace std