#pragma once

#include "vuk/Hash.hpp"
#include <string_view>

namespace vuk {
	class Name {
	public:
		Name() = default;

		Name(decltype(nullptr));
		Name(const char* str) noexcept;
		Name(std::string_view str) noexcept;

		std::string_view to_sv() const noexcept;
		const char* c_str() const noexcept {
			return id;
		}

		Name append(std::string_view other) const noexcept;

		bool is_invalid() const noexcept;

		friend bool operator==(Name a, Name b) noexcept {
			return a.id == b.id;
		}

		friend bool operator!=(Name a, Name b) noexcept {
			return a.id != b.id;
		}

		friend bool operator<(Name a, Name b) noexcept {
			return (uintptr_t)a.id < (uintptr_t)b.id;
		}

	private:
		static constexpr const char invalid_value[] = "INVALID";
		const char* id = invalid_value;

		friend struct std::hash<vuk::Name>;
	};

	struct QualifiedName {
		Name prefix;
		Name name;

		bool operator==(const QualifiedName&) const noexcept = default;
		bool is_invalid() const noexcept {
			return name.is_invalid();
		}
	};
} // namespace vuk

namespace std {
	template<>
	struct hash<vuk::Name> {
		size_t operator()(vuk::Name const& s) const;
	};

	template<>
	struct hash<vuk::QualifiedName> {
		size_t operator()(vuk::QualifiedName const& s) const;
	};
} // namespace std
