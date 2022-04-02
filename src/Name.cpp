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
			auto hash = hash::fnv1a::hash(s.data(), (uint32_t)s.size());
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
				} else {
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

	Name Name::append(Name other) const noexcept {
		std::string app;
		app.reserve(strlen(id) + strlen(other.id));
		app.append(id);
		app.append(other.id);
		return Name(app);
	}
} // namespace vuk

namespace std {
	size_t hash<vuk::Name>::operator()(vuk::Name const& s) const {
		return hash<const char*>()(s.id);
	}
} // namespace std