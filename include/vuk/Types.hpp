#pragma once

#include "vuk/Config.hpp"
#include "vuk/Flags.hpp"
#include "vuk/Hash.hpp"
#include "vuk/vuk_fwd.hpp"

#include <compare>
#include <string>
#include <string_view>
#include <type_traits>

#define MOV(x) (static_cast<std::remove_reference_t<decltype(x)>&&>(x))

namespace vuk {
	struct HandleBase {
		size_t id = UINT64_MAX;
	};

	template<class T>
	struct Handle : public HandleBase {
		T payload;

		constexpr bool operator==(const Handle& o) const noexcept {
			return id == o.id;
		}
	};

	template<typename Type>
	class Unique {
		Allocator* allocator;
		Type payload;

	public:
		using element_type = Type;

		explicit Unique() : allocator(nullptr), payload{} {}
		explicit Unique(Allocator& allocator) : allocator(&allocator), payload{} {}
		explicit Unique(Allocator& allocator, Type payload) : allocator(&allocator), payload(MOV(payload)) {}
		Unique(Unique const&) = delete;

		Unique(Unique&& other) noexcept : allocator(other.allocator), payload(other.release()) {}

		~Unique() noexcept;

		Unique& operator=(Unique const&) = delete;

		Unique& operator=(Unique&& other) noexcept {
			auto tmp = other.allocator;
			reset(other.release());
			allocator = tmp;
			return *this;
		}

		explicit operator bool() const noexcept {
			return static_cast<bool>(payload);
		}

		Type const* operator->() const noexcept {
			return &payload;
		}

		Type* operator->() noexcept {
			return &payload;
		}

		Type const& operator*() const noexcept {
			return payload;
		}

		Type& operator*() noexcept {
			return payload;
		}

		const Type& get() const noexcept {
			return payload;
		}

		Type& get() noexcept {
			return payload;
		}

		void reset(Type value = Type()) noexcept;

		Type release() noexcept {
			allocator = nullptr;
			return MOV(payload);
		}

		void swap(Unique<Type>& rhs) noexcept {
			std::swap(payload, rhs.payload);
			std::swap(allocator, rhs.allocator);
		}
	};

	template<typename Type>
	inline void swap(Unique<Type>& lhs, Unique<Type>& rhs) noexcept {
		lhs.swap(rhs);
	}
} // namespace vuk

namespace std {
	template<class T>
	struct hash<vuk::Handle<T>> {
		size_t operator()(vuk::Handle<T> const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.id, reinterpret_cast<uint64_t>(x.payload));
			return h;
		}
	};
} // namespace std

namespace vuk {
	enum class SampleCountFlagBits : VkSampleCountFlags {
		e1 = VK_SAMPLE_COUNT_1_BIT,
		e2 = VK_SAMPLE_COUNT_2_BIT,
		e4 = VK_SAMPLE_COUNT_4_BIT,
		e8 = VK_SAMPLE_COUNT_8_BIT,
		e16 = VK_SAMPLE_COUNT_16_BIT,
		e32 = VK_SAMPLE_COUNT_32_BIT,
		e64 = VK_SAMPLE_COUNT_64_BIT,
		eInfer = 1024
	};

	struct Samples {
		SampleCountFlagBits count;

		struct Framebuffer {};

		Samples() noexcept : count(SampleCountFlagBits::e1) {}
		Samples(SampleCountFlagBits samples) noexcept : count(samples) {}
		Samples(Framebuffer) noexcept : count(SampleCountFlagBits::eInfer) {}

		constexpr static auto e1 = SampleCountFlagBits::e1;
		constexpr static auto e2 = SampleCountFlagBits::e2;
		constexpr static auto e4 = SampleCountFlagBits::e4;
		constexpr static auto e8 = SampleCountFlagBits::e8;
		constexpr static auto e16 = SampleCountFlagBits::e16;
		constexpr static auto e32 = SampleCountFlagBits::e32;
		constexpr static auto e64 = SampleCountFlagBits::e64;
		constexpr static auto eInfer = SampleCountFlagBits::eInfer;
	};

	inline bool operator==(Samples const& a, Samples const& b) noexcept {
		return a.count == b.count;
	}

	struct Offset3D;
	struct Offset2D {
		int32_t x = {};
		int32_t y = {};

		bool operator==(Offset2D const& rhs) const noexcept {
			return (x == rhs.x) && (y == rhs.y);
		}

		bool operator!=(Offset2D const& rhs) const noexcept {
			return !operator==(rhs);
		}

		operator VkOffset2D const&() const noexcept {
			return *reinterpret_cast<const VkOffset2D*>(this);
		}

		operator VkOffset2D&() noexcept {
			return *reinterpret_cast<VkOffset2D*>(this);
		}

		explicit operator Offset3D();
	};

	struct Extent3D;
	struct Extent2D {
		uint32_t width = {};
		uint32_t height = {};

		bool operator==(Extent2D const& rhs) const noexcept {
			return (width == rhs.width) && (height == rhs.height);
		}

		bool operator!=(Extent2D const& rhs) const noexcept {
			return !operator==(rhs);
		}

		operator VkExtent2D const&() const noexcept {
			return *reinterpret_cast<const VkExtent2D*>(this);
		}

		operator VkExtent2D&() noexcept {
			return *reinterpret_cast<VkExtent2D*>(this);
		}

		explicit operator Extent3D();
	};

	struct Offset3D {
		int32_t x = {};
		int32_t y = {};
		int32_t z = {};

		bool operator==(Offset3D const& rhs) const noexcept {
			return (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
		}

		bool operator!=(Offset3D const& rhs) const noexcept {
			return !operator==(rhs);
		}

		operator VkOffset3D const&() const noexcept {
			return *reinterpret_cast<const VkOffset3D*>(this);
		}

		operator VkOffset3D&() noexcept {
			return *reinterpret_cast<VkOffset3D*>(this);
		}
	};

	inline Offset2D::operator Offset3D() {
		return Offset3D{ x, y, 0 };
	}

	struct Extent3D {
		uint32_t width = {};
		uint32_t height = {};
		uint32_t depth = {};

		auto operator<=>(const Extent3D&) const = default;

		operator VkExtent3D const&() const noexcept {
			return *reinterpret_cast<const VkExtent3D*>(this);
		}

		operator VkExtent3D&() noexcept {
			return *reinterpret_cast<VkExtent3D*>(this);
		}
	};

	inline Extent2D::operator Extent3D() {
		return Extent3D{ width, height, 1u };
	}

	struct Viewport {
		float x = {};
		float y = {};
		float width = {};
		float height = {};
		float minDepth = 0.f;
		float maxDepth = 1.f;

		operator VkViewport const&() const noexcept {
			return *reinterpret_cast<const VkViewport*>(this);
		}

		operator VkViewport&() noexcept {
			return *reinterpret_cast<VkViewport*>(this);
		}

		bool operator==(Viewport const& rhs) const noexcept {
			return (x == rhs.x) && (y == rhs.y) && (width == rhs.width) && (height == rhs.height) && (minDepth == rhs.minDepth) && (maxDepth == rhs.maxDepth);
		}

		bool operator!=(Viewport const& rhs) const noexcept {
			return !operator==(rhs);
		}
	};
	static_assert(sizeof(Viewport) == sizeof(VkViewport), "struct and wrapper have different size!");
	static_assert(std::is_standard_layout<Viewport>::value, "struct wrapper is not a standard layout!");

	enum class Sizing { eAbsolute, eRelative };

	struct Rect2D {
		Sizing sizing = Sizing::eAbsolute;

		Offset2D offset = {};
		Extent2D extent = {};

		struct {
			float x = 0.f;
			float y = 0.f;
			float width = 1.0f;
			float height = 1.0f;
		} _relative;

		static Rect2D absolute(int32_t x, int32_t y, uint32_t width, uint32_t height) noexcept {
			return Rect2D{ .offset = { x, y }, .extent = { width, height } };
		}
		static Rect2D absolute(Offset2D offset, Extent2D extent) noexcept {
			return Rect2D{ .offset = offset, .extent = extent };
		}
		static Rect2D relative(float x, float y, float width, float height) noexcept {
			return Rect2D{ .sizing = Sizing::eRelative, ._relative = { .x = x, .y = y, .width = width, .height = height } };
		}
		static Rect2D framebuffer() noexcept {
			return Rect2D{ .sizing = Sizing::eRelative };
		}
	};

	enum class Format {
		eUndefined = VK_FORMAT_UNDEFINED,
		eR4G4UnormPack8 = VK_FORMAT_R4G4_UNORM_PACK8,
		eR4G4B4A4UnormPack16 = VK_FORMAT_R4G4B4A4_UNORM_PACK16,
		eB4G4R4A4UnormPack16 = VK_FORMAT_B4G4R4A4_UNORM_PACK16,
		eR5G6B5UnormPack16 = VK_FORMAT_R5G6B5_UNORM_PACK16,
		eB5G6R5UnormPack16 = VK_FORMAT_B5G6R5_UNORM_PACK16,
		eR5G5B5A1UnormPack16 = VK_FORMAT_R5G5B5A1_UNORM_PACK16,
		eB5G5R5A1UnormPack16 = VK_FORMAT_B5G5R5A1_UNORM_PACK16,
		eA1R5G5B5UnormPack16 = VK_FORMAT_A1R5G5B5_UNORM_PACK16,
		eR8Unorm = VK_FORMAT_R8_UNORM,
		eR8Snorm = VK_FORMAT_R8_SNORM,
		eR8Uscaled = VK_FORMAT_R8_USCALED,
		eR8Sscaled = VK_FORMAT_R8_SSCALED,
		eR8Uint = VK_FORMAT_R8_UINT,
		eR8Sint = VK_FORMAT_R8_SINT,
		eR8Srgb = VK_FORMAT_R8_SRGB,
		eR8G8Unorm = VK_FORMAT_R8G8_UNORM,
		eR8G8Snorm = VK_FORMAT_R8G8_SNORM,
		eR8G8Uscaled = VK_FORMAT_R8G8_USCALED,
		eR8G8Sscaled = VK_FORMAT_R8G8_SSCALED,
		eR8G8Uint = VK_FORMAT_R8G8_UINT,
		eR8G8Sint = VK_FORMAT_R8G8_SINT,
		eR8G8Srgb = VK_FORMAT_R8G8_SRGB,
		eR8G8B8Unorm = VK_FORMAT_R8G8B8_UNORM,
		eR8G8B8Snorm = VK_FORMAT_R8G8B8_SNORM,
		eR8G8B8Uscaled = VK_FORMAT_R8G8B8_USCALED,
		eR8G8B8Sscaled = VK_FORMAT_R8G8B8_SSCALED,
		eR8G8B8Uint = VK_FORMAT_R8G8B8_UINT,
		eR8G8B8Sint = VK_FORMAT_R8G8B8_SINT,
		eR8G8B8Srgb = VK_FORMAT_R8G8B8_SRGB,
		eB8G8R8Unorm = VK_FORMAT_B8G8R8_UNORM,
		eB8G8R8Snorm = VK_FORMAT_B8G8R8_SNORM,
		eB8G8R8Uscaled = VK_FORMAT_B8G8R8_USCALED,
		eB8G8R8Sscaled = VK_FORMAT_B8G8R8_SSCALED,
		eB8G8R8Uint = VK_FORMAT_B8G8R8_UINT,
		eB8G8R8Sint = VK_FORMAT_B8G8R8_SINT,
		eB8G8R8Srgb = VK_FORMAT_B8G8R8_SRGB,
		eR8G8B8A8Unorm = VK_FORMAT_R8G8B8A8_UNORM,
		eR8G8B8A8Snorm = VK_FORMAT_R8G8B8A8_SNORM,
		eR8G8B8A8Uscaled = VK_FORMAT_R8G8B8A8_USCALED,
		eR8G8B8A8Sscaled = VK_FORMAT_R8G8B8A8_SSCALED,
		eR8G8B8A8Uint = VK_FORMAT_R8G8B8A8_UINT,
		eR8G8B8A8Sint = VK_FORMAT_R8G8B8A8_SINT,
		eR8G8B8A8Srgb = VK_FORMAT_R8G8B8A8_SRGB,
		eB8G8R8A8Unorm = VK_FORMAT_B8G8R8A8_UNORM,
		eB8G8R8A8Snorm = VK_FORMAT_B8G8R8A8_SNORM,
		eB8G8R8A8Uscaled = VK_FORMAT_B8G8R8A8_USCALED,
		eB8G8R8A8Sscaled = VK_FORMAT_B8G8R8A8_SSCALED,
		eB8G8R8A8Uint = VK_FORMAT_B8G8R8A8_UINT,
		eB8G8R8A8Sint = VK_FORMAT_B8G8R8A8_SINT,
		eB8G8R8A8Srgb = VK_FORMAT_B8G8R8A8_SRGB,
		eA8B8G8R8UnormPack32 = VK_FORMAT_A8B8G8R8_UNORM_PACK32,
		eA8B8G8R8SnormPack32 = VK_FORMAT_A8B8G8R8_SNORM_PACK32,
		eA8B8G8R8UscaledPack32 = VK_FORMAT_A8B8G8R8_USCALED_PACK32,
		eA8B8G8R8SscaledPack32 = VK_FORMAT_A8B8G8R8_SSCALED_PACK32,
		eA8B8G8R8UintPack32 = VK_FORMAT_A8B8G8R8_UINT_PACK32,
		eA8B8G8R8SintPack32 = VK_FORMAT_A8B8G8R8_SINT_PACK32,
		eA8B8G8R8SrgbPack32 = VK_FORMAT_A8B8G8R8_SRGB_PACK32,
		eA2R10G10B10UnormPack32 = VK_FORMAT_A2R10G10B10_UNORM_PACK32,
		eA2R10G10B10SnormPack32 = VK_FORMAT_A2R10G10B10_SNORM_PACK32,
		eA2R10G10B10UscaledPack32 = VK_FORMAT_A2R10G10B10_USCALED_PACK32,
		eA2R10G10B10SscaledPack32 = VK_FORMAT_A2R10G10B10_SSCALED_PACK32,
		eA2R10G10B10UintPack32 = VK_FORMAT_A2R10G10B10_UINT_PACK32,
		eA2R10G10B10SintPack32 = VK_FORMAT_A2R10G10B10_SINT_PACK32,
		eA2B10G10R10UnormPack32 = VK_FORMAT_A2B10G10R10_UNORM_PACK32,
		eA2B10G10R10SnormPack32 = VK_FORMAT_A2B10G10R10_SNORM_PACK32,
		eA2B10G10R10UscaledPack32 = VK_FORMAT_A2B10G10R10_USCALED_PACK32,
		eA2B10G10R10SscaledPack32 = VK_FORMAT_A2B10G10R10_SSCALED_PACK32,
		eA2B10G10R10UintPack32 = VK_FORMAT_A2B10G10R10_UINT_PACK32,
		eA2B10G10R10SintPack32 = VK_FORMAT_A2B10G10R10_SINT_PACK32,
		eR16Unorm = VK_FORMAT_R16_UNORM,
		eR16Snorm = VK_FORMAT_R16_SNORM,
		eR16Uscaled = VK_FORMAT_R16_USCALED,
		eR16Sscaled = VK_FORMAT_R16_SSCALED,
		eR16Uint = VK_FORMAT_R16_UINT,
		eR16Sint = VK_FORMAT_R16_SINT,
		eR16Sfloat = VK_FORMAT_R16_SFLOAT,
		eR16G16Unorm = VK_FORMAT_R16G16_UNORM,
		eR16G16Snorm = VK_FORMAT_R16G16_SNORM,
		eR16G16Uscaled = VK_FORMAT_R16G16_USCALED,
		eR16G16Sscaled = VK_FORMAT_R16G16_SSCALED,
		eR16G16Uint = VK_FORMAT_R16G16_UINT,
		eR16G16Sint = VK_FORMAT_R16G16_SINT,
		eR16G16Sfloat = VK_FORMAT_R16G16_SFLOAT,
		eR16G16B16Unorm = VK_FORMAT_R16G16B16_UNORM,
		eR16G16B16Snorm = VK_FORMAT_R16G16B16_SNORM,
		eR16G16B16Uscaled = VK_FORMAT_R16G16B16_USCALED,
		eR16G16B16Sscaled = VK_FORMAT_R16G16B16_SSCALED,
		eR16G16B16Uint = VK_FORMAT_R16G16B16_UINT,
		eR16G16B16Sint = VK_FORMAT_R16G16B16_SINT,
		eR16G16B16Sfloat = VK_FORMAT_R16G16B16_SFLOAT,
		eR16G16B16A16Unorm = VK_FORMAT_R16G16B16A16_UNORM,
		eR16G16B16A16Snorm = VK_FORMAT_R16G16B16A16_SNORM,
		eR16G16B16A16Uscaled = VK_FORMAT_R16G16B16A16_USCALED,
		eR16G16B16A16Sscaled = VK_FORMAT_R16G16B16A16_SSCALED,
		eR16G16B16A16Uint = VK_FORMAT_R16G16B16A16_UINT,
		eR16G16B16A16Sint = VK_FORMAT_R16G16B16A16_SINT,
		eR16G16B16A16Sfloat = VK_FORMAT_R16G16B16A16_SFLOAT,
		eR32Uint = VK_FORMAT_R32_UINT,
		eR32Sint = VK_FORMAT_R32_SINT,
		eR32Sfloat = VK_FORMAT_R32_SFLOAT,
		eR32G32Uint = VK_FORMAT_R32G32_UINT,
		eR32G32Sint = VK_FORMAT_R32G32_SINT,
		eR32G32Sfloat = VK_FORMAT_R32G32_SFLOAT,
		eR32G32B32Uint = VK_FORMAT_R32G32B32_UINT,
		eR32G32B32Sint = VK_FORMAT_R32G32B32_SINT,
		eR32G32B32Sfloat = VK_FORMAT_R32G32B32_SFLOAT,
		eR32G32B32A32Uint = VK_FORMAT_R32G32B32A32_UINT,
		eR32G32B32A32Sint = VK_FORMAT_R32G32B32A32_SINT,
		eR32G32B32A32Sfloat = VK_FORMAT_R32G32B32A32_SFLOAT,
		eR64Uint = VK_FORMAT_R64_UINT,
		eR64Sint = VK_FORMAT_R64_SINT,
		eR64Sfloat = VK_FORMAT_R64_SFLOAT,
		eR64G64Uint = VK_FORMAT_R64G64_UINT,
		eR64G64Sint = VK_FORMAT_R64G64_SINT,
		eR64G64Sfloat = VK_FORMAT_R64G64_SFLOAT,
		eR64G64B64Uint = VK_FORMAT_R64G64B64_UINT,
		eR64G64B64Sint = VK_FORMAT_R64G64B64_SINT,
		eR64G64B64Sfloat = VK_FORMAT_R64G64B64_SFLOAT,
		eR64G64B64A64Uint = VK_FORMAT_R64G64B64A64_UINT,
		eR64G64B64A64Sint = VK_FORMAT_R64G64B64A64_SINT,
		eR64G64B64A64Sfloat = VK_FORMAT_R64G64B64A64_SFLOAT,
		eB10G11R11UfloatPack32 = VK_FORMAT_B10G11R11_UFLOAT_PACK32,
		eE5B9G9R9UfloatPack32 = VK_FORMAT_E5B9G9R9_UFLOAT_PACK32,
		eD16Unorm = VK_FORMAT_D16_UNORM,
		eX8D24UnormPack32 = VK_FORMAT_X8_D24_UNORM_PACK32,
		eD32Sfloat = VK_FORMAT_D32_SFLOAT,
		eS8Uint = VK_FORMAT_S8_UINT,
		eD16UnormS8Uint = VK_FORMAT_D16_UNORM_S8_UINT,
		eD24UnormS8Uint = VK_FORMAT_D24_UNORM_S8_UINT,
		eD32SfloatS8Uint = VK_FORMAT_D32_SFLOAT_S8_UINT,
		eBc1RgbUnormBlock = VK_FORMAT_BC1_RGB_UNORM_BLOCK,
		eBc1RgbSrgbBlock = VK_FORMAT_BC1_RGB_SRGB_BLOCK,
		eBc1RgbaUnormBlock = VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
		eBc1RgbaSrgbBlock = VK_FORMAT_BC1_RGBA_SRGB_BLOCK,
		eBc2UnormBlock = VK_FORMAT_BC2_UNORM_BLOCK,
		eBc2SrgbBlock = VK_FORMAT_BC2_SRGB_BLOCK,
		eBc3UnormBlock = VK_FORMAT_BC3_UNORM_BLOCK,
		eBc3SrgbBlock = VK_FORMAT_BC3_SRGB_BLOCK,
		eBc4UnormBlock = VK_FORMAT_BC4_UNORM_BLOCK,
		eBc4SnormBlock = VK_FORMAT_BC4_SNORM_BLOCK,
		eBc5UnormBlock = VK_FORMAT_BC5_UNORM_BLOCK,
		eBc5SnormBlock = VK_FORMAT_BC5_SNORM_BLOCK,
		eBc6HUfloatBlock = VK_FORMAT_BC6H_UFLOAT_BLOCK,
		eBc6HSfloatBlock = VK_FORMAT_BC6H_SFLOAT_BLOCK,
		eBc7UnormBlock = VK_FORMAT_BC7_UNORM_BLOCK,
		eBc7SrgbBlock = VK_FORMAT_BC7_SRGB_BLOCK,
		eEtc2R8G8B8UnormBlock = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK,
		eEtc2R8G8B8SrgbBlock = VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK,
		eEtc2R8G8B8A1UnormBlock = VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK,
		eEtc2R8G8B8A1SrgbBlock = VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK,
		eEtc2R8G8B8A8UnormBlock = VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK,
		eEtc2R8G8B8A8SrgbBlock = VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
		eEacR11UnormBlock = VK_FORMAT_EAC_R11_UNORM_BLOCK,
		eEacR11SnormBlock = VK_FORMAT_EAC_R11_SNORM_BLOCK,
		eEacR11G11UnormBlock = VK_FORMAT_EAC_R11G11_UNORM_BLOCK,
		eEacR11G11SnormBlock = VK_FORMAT_EAC_R11G11_SNORM_BLOCK,
		eAstc4x4UnormBlock = VK_FORMAT_ASTC_4x4_UNORM_BLOCK,
		eAstc4x4SrgbBlock = VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
		eAstc5x4UnormBlock = VK_FORMAT_ASTC_5x4_UNORM_BLOCK,
		eAstc5x4SrgbBlock = VK_FORMAT_ASTC_5x4_SRGB_BLOCK,
		eAstc5x5UnormBlock = VK_FORMAT_ASTC_5x5_UNORM_BLOCK,
		eAstc5x5SrgbBlock = VK_FORMAT_ASTC_5x5_SRGB_BLOCK,
		eAstc6x5UnormBlock = VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
		eAstc6x5SrgbBlock = VK_FORMAT_ASTC_6x5_SRGB_BLOCK,
		eAstc6x6UnormBlock = VK_FORMAT_ASTC_6x6_UNORM_BLOCK,
		eAstc6x6SrgbBlock = VK_FORMAT_ASTC_6x6_SRGB_BLOCK,
		eAstc8x5UnormBlock = VK_FORMAT_ASTC_8x5_UNORM_BLOCK,
		eAstc8x5SrgbBlock = VK_FORMAT_ASTC_8x5_SRGB_BLOCK,
		eAstc8x6UnormBlock = VK_FORMAT_ASTC_8x6_UNORM_BLOCK,
		eAstc8x6SrgbBlock = VK_FORMAT_ASTC_8x6_SRGB_BLOCK,
		eAstc8x8UnormBlock = VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
		eAstc8x8SrgbBlock = VK_FORMAT_ASTC_8x8_SRGB_BLOCK,
		eAstc10x5UnormBlock = VK_FORMAT_ASTC_10x5_UNORM_BLOCK,
		eAstc10x5SrgbBlock = VK_FORMAT_ASTC_10x5_SRGB_BLOCK,
		eAstc10x6UnormBlock = VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
		eAstc10x6SrgbBlock = VK_FORMAT_ASTC_10x6_SRGB_BLOCK,
		eAstc10x8UnormBlock = VK_FORMAT_ASTC_10x8_UNORM_BLOCK,
		eAstc10x8SrgbBlock = VK_FORMAT_ASTC_10x8_SRGB_BLOCK,
		eAstc10x10UnormBlock = VK_FORMAT_ASTC_10x10_UNORM_BLOCK,
		eAstc10x10SrgbBlock = VK_FORMAT_ASTC_10x10_SRGB_BLOCK,
		eAstc12x10UnormBlock = VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
		eAstc12x10SrgbBlock = VK_FORMAT_ASTC_12x10_SRGB_BLOCK,
		eAstc12x12UnormBlock = VK_FORMAT_ASTC_12x12_UNORM_BLOCK,
		eAstc12x12SrgbBlock = VK_FORMAT_ASTC_12x12_SRGB_BLOCK,
		eG8B8G8R8422Unorm = VK_FORMAT_G8B8G8R8_422_UNORM,
		eB8G8R8G8422Unorm = VK_FORMAT_B8G8R8G8_422_UNORM,
		eG8B8R83Plane420Unorm = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
		eG8B8R82Plane420Unorm = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM,
		eG8B8R83Plane422Unorm = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM,
		eG8B8R82Plane422Unorm = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM,
		eG8B8R83Plane444Unorm = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM,
		eR10X6UnormPack16 = VK_FORMAT_R10X6_UNORM_PACK16,
		eR10X6G10X6Unorm2Pack16 = VK_FORMAT_R10X6G10X6_UNORM_2PACK16,
		eR10X6G10X6B10X6A10X6Unorm4Pack16 = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16,
		eG10X6B10X6G10X6R10X6422Unorm4Pack16 = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16,
		eB10X6G10X6R10X6G10X6422Unorm4Pack16 = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16,
		eG10X6B10X6R10X63Plane420Unorm3Pack16 = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16,
		eG10X6B10X6R10X62Plane420Unorm3Pack16 = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
		eG10X6B10X6R10X63Plane422Unorm3Pack16 = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16,
		eG10X6B10X6R10X62Plane422Unorm3Pack16 = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16,
		eG10X6B10X6R10X63Plane444Unorm3Pack16 = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16,
		eR12X4UnormPack16 = VK_FORMAT_R12X4_UNORM_PACK16,
		eR12X4G12X4Unorm2Pack16 = VK_FORMAT_R12X4G12X4_UNORM_2PACK16,
		eR12X4G12X4B12X4A12X4Unorm4Pack16 = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16,
		eG12X4B12X4G12X4R12X4422Unorm4Pack16 = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16,
		eB12X4G12X4R12X4G12X4422Unorm4Pack16 = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16,
		eG12X4B12X4R12X43Plane420Unorm3Pack16 = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16,
		eG12X4B12X4R12X42Plane420Unorm3Pack16 = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16,
		eG12X4B12X4R12X43Plane422Unorm3Pack16 = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16,
		eG12X4B12X4R12X42Plane422Unorm3Pack16 = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16,
		eG12X4B12X4R12X43Plane444Unorm3Pack16 = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16,
		eG16B16G16R16422Unorm = VK_FORMAT_G16B16G16R16_422_UNORM,
		eB16G16R16G16422Unorm = VK_FORMAT_B16G16R16G16_422_UNORM,
		eG16B16R163Plane420Unorm = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM,
		eG16B16R162Plane420Unorm = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM,
		eG16B16R163Plane422Unorm = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM,
		eG16B16R162Plane422Unorm = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM,
		eG16B16R163Plane444Unorm = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM,
		ePvrtc12BppUnormBlockIMG = VK_FORMAT_PVRTC1_2BPP_UNORM_BLOCK_IMG,
		ePvrtc14BppUnormBlockIMG = VK_FORMAT_PVRTC1_4BPP_UNORM_BLOCK_IMG,
		ePvrtc22BppUnormBlockIMG = VK_FORMAT_PVRTC2_2BPP_UNORM_BLOCK_IMG,
		ePvrtc24BppUnormBlockIMG = VK_FORMAT_PVRTC2_4BPP_UNORM_BLOCK_IMG,
		ePvrtc12BppSrgbBlockIMG = VK_FORMAT_PVRTC1_2BPP_SRGB_BLOCK_IMG,
		ePvrtc14BppSrgbBlockIMG = VK_FORMAT_PVRTC1_4BPP_SRGB_BLOCK_IMG,
		ePvrtc22BppSrgbBlockIMG = VK_FORMAT_PVRTC2_2BPP_SRGB_BLOCK_IMG,
		ePvrtc24BppSrgbBlockIMG = VK_FORMAT_PVRTC2_4BPP_SRGB_BLOCK_IMG,
		eAstc4x4SfloatBlockEXT = VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT,
		eAstc5x4SfloatBlockEXT = VK_FORMAT_ASTC_5x4_SFLOAT_BLOCK_EXT,
		eAstc5x5SfloatBlockEXT = VK_FORMAT_ASTC_5x5_SFLOAT_BLOCK_EXT,
		eAstc6x5SfloatBlockEXT = VK_FORMAT_ASTC_6x5_SFLOAT_BLOCK_EXT,
		eAstc6x6SfloatBlockEXT = VK_FORMAT_ASTC_6x6_SFLOAT_BLOCK_EXT,
		eAstc8x5SfloatBlockEXT = VK_FORMAT_ASTC_8x5_SFLOAT_BLOCK_EXT,
		eAstc8x6SfloatBlockEXT = VK_FORMAT_ASTC_8x6_SFLOAT_BLOCK_EXT,
		eAstc8x8SfloatBlockEXT = VK_FORMAT_ASTC_8x8_SFLOAT_BLOCK_EXT,
		eAstc10x5SfloatBlockEXT = VK_FORMAT_ASTC_10x5_SFLOAT_BLOCK_EXT,
		eAstc10x6SfloatBlockEXT = VK_FORMAT_ASTC_10x6_SFLOAT_BLOCK_EXT,
		eAstc10x8SfloatBlockEXT = VK_FORMAT_ASTC_10x8_SFLOAT_BLOCK_EXT,
		eAstc10x10SfloatBlockEXT = VK_FORMAT_ASTC_10x10_SFLOAT_BLOCK_EXT,
		eAstc12x10SfloatBlockEXT = VK_FORMAT_ASTC_12x10_SFLOAT_BLOCK_EXT,
		eAstc12x12SfloatBlockEXT = VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT,
		eB10X6G10X6R10X6G10X6422Unorm4Pack16KHR = VK_FORMAT_B10X6G10X6R10X6G10X6_422_UNORM_4PACK16_KHR,
		eB12X4G12X4R12X4G12X4422Unorm4Pack16KHR = VK_FORMAT_B12X4G12X4R12X4G12X4_422_UNORM_4PACK16_KHR,
		eB16G16R16G16422UnormKHR = VK_FORMAT_B16G16R16G16_422_UNORM_KHR,
		eB8G8R8G8422UnormKHR = VK_FORMAT_B8G8R8G8_422_UNORM_KHR,
		eG10X6B10X6G10X6R10X6422Unorm4Pack16KHR = VK_FORMAT_G10X6B10X6G10X6R10X6_422_UNORM_4PACK16_KHR,
		eG10X6B10X6R10X62Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16_KHR,
		eG10X6B10X6R10X62Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6R10X6_2PLANE_422_UNORM_3PACK16_KHR,
		eG10X6B10X6R10X63Plane420Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_420_UNORM_3PACK16_KHR,
		eG10X6B10X6R10X63Plane422Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_422_UNORM_3PACK16_KHR,
		eG10X6B10X6R10X63Plane444Unorm3Pack16KHR = VK_FORMAT_G10X6_B10X6_R10X6_3PLANE_444_UNORM_3PACK16_KHR,
		eG12X4B12X4G12X4R12X4422Unorm4Pack16KHR = VK_FORMAT_G12X4B12X4G12X4R12X4_422_UNORM_4PACK16_KHR,
		eG12X4B12X4R12X42Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_420_UNORM_3PACK16_KHR,
		eG12X4B12X4R12X42Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4R12X4_2PLANE_422_UNORM_3PACK16_KHR,
		eG12X4B12X4R12X43Plane420Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_420_UNORM_3PACK16_KHR,
		eG12X4B12X4R12X43Plane422Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_422_UNORM_3PACK16_KHR,
		eG12X4B12X4R12X43Plane444Unorm3Pack16KHR = VK_FORMAT_G12X4_B12X4_R12X4_3PLANE_444_UNORM_3PACK16_KHR,
		eG16B16G16R16422UnormKHR = VK_FORMAT_G16B16G16R16_422_UNORM_KHR,
		eG16B16R162Plane420UnormKHR = VK_FORMAT_G16_B16R16_2PLANE_420_UNORM_KHR,
		eG16B16R162Plane422UnormKHR = VK_FORMAT_G16_B16R16_2PLANE_422_UNORM_KHR,
		eG16B16R163Plane420UnormKHR = VK_FORMAT_G16_B16_R16_3PLANE_420_UNORM_KHR,
		eG16B16R163Plane422UnormKHR = VK_FORMAT_G16_B16_R16_3PLANE_422_UNORM_KHR,
		eG16B16R163Plane444UnormKHR = VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM_KHR,
		eG8B8G8R8422UnormKHR = VK_FORMAT_G8B8G8R8_422_UNORM_KHR,
		eG8B8R82Plane420UnormKHR = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM_KHR,
		eG8B8R82Plane422UnormKHR = VK_FORMAT_G8_B8R8_2PLANE_422_UNORM_KHR,
		eG8B8R83Plane420UnormKHR = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM_KHR,
		eG8B8R83Plane422UnormKHR = VK_FORMAT_G8_B8_R8_3PLANE_422_UNORM_KHR,
		eG8B8R83Plane444UnormKHR = VK_FORMAT_G8_B8_R8_3PLANE_444_UNORM_KHR,
		eR10X6G10X6B10X6A10X6Unorm4Pack16KHR = VK_FORMAT_R10X6G10X6B10X6A10X6_UNORM_4PACK16_KHR,
		eR10X6G10X6Unorm2Pack16KHR = VK_FORMAT_R10X6G10X6_UNORM_2PACK16_KHR,
		eR10X6UnormPack16KHR = VK_FORMAT_R10X6_UNORM_PACK16_KHR,
		eR12X4G12X4B12X4A12X4Unorm4Pack16KHR = VK_FORMAT_R12X4G12X4B12X4A12X4_UNORM_4PACK16_KHR,
		eR12X4G12X4Unorm2Pack16KHR = VK_FORMAT_R12X4G12X4_UNORM_2PACK16_KHR,
		eR12X4UnormPack16KHR = VK_FORMAT_R12X4_UNORM_PACK16_KHR
	};

	// return the texel block size of a format
	uint32_t format_to_texel_block_size(Format) noexcept;
	// return the 3D texel block extent of a format
	Extent3D format_to_texel_block_extent(Format) noexcept;
	// compute the byte size of an image with given format and extent
	uint32_t compute_image_size(Format, Extent3D) noexcept;
	// get name of format
	std::string_view format_to_sv(Format format) noexcept;
	// true if format performs automatic sRGB conversion
	bool is_format_srgb(Format) noexcept;
	// get the unorm equivalent of the srgb format (returns Format::Undefined if the format doesn't exist)
	Format unorm_to_srgb(Format) noexcept;
	// get the srgb equivalent of the unorm format (returns Format::Undefined if the format doesn't exist)
	Format srgb_to_unorm(Format) noexcept;

	enum class MemoryUsage {
		eGPUonly = 1 /*VMA_MEMORY_USAGE_GPU_ONLY*/,
		eCPUtoGPU = 3 /*VMA_MEMORY_USAGE_CPU_TO_GPU*/,
		eCPUonly = 2 /*VMA_MEMORY_USAGE_CPU_ONLY*/,
		eGPUtoCPU = 4 /*VMA_MEMORY_USAGE_GPU_TO_CPU*/
	};

	using Bool32 = uint32_t;
	struct ClearColor {
		constexpr ClearColor(uint32_t r, uint32_t g, uint32_t b, uint32_t a) noexcept {
			ccv.uint32[0] = r;
			ccv.uint32[1] = g;
			ccv.uint32[2] = b;
			ccv.uint32[3] = a;
		}

		constexpr ClearColor(int32_t r, int32_t g, int32_t b, int32_t a) noexcept {
			ccv.int32[0] = r;
			ccv.int32[1] = g;
			ccv.int32[2] = b;
			ccv.int32[3] = a;
		}

		constexpr ClearColor(float r, float g, float b, float a) noexcept {
			ccv.float32[0] = r;
			ccv.float32[1] = g;
			ccv.float32[2] = b;
			ccv.float32[3] = a;
		}
		VkClearColorValue ccv;
	};

	template<class T>
	static constexpr ClearColor White = { 1, 1, 1, 1 };
	template<class T>
	static constexpr ClearColor Black = { 0, 0, 0, 1 };
	template<class T>
	static constexpr ClearColor Transparent = { 0, 0, 0, 0 };

	template<>
	inline constexpr ClearColor White<float> = { 1.f, 1.f, 1.f, 1.f };
	template<>
	inline constexpr ClearColor White<unsigned> = { 1u, 1u, 1u, 1u };
	template<>
	inline constexpr ClearColor White<signed> = { 1, 1, 1, 1 };

	template<>
	inline constexpr ClearColor Black<float> = { 0.f, 0.f, 0.f, 1.f };
	template<>
	inline constexpr ClearColor Black<unsigned> = { 0u, 0u, 0u, 1u };
	template<>
	inline constexpr ClearColor Black<signed> = { 0, 0, 0, 1 };

	template<>
	inline constexpr ClearColor Transparent<float> = { 0.f, 0.f, 0.f, 0.f };
	template<>
	inline constexpr ClearColor Transparent<unsigned> = { 0u, 0u, 0u, 0u };
	template<>
	inline constexpr ClearColor Transparent<signed> = { 0, 0, 0, 0 };

	struct ClearDepthStencil {
		constexpr ClearDepthStencil(float depth, uint32_t stencil) noexcept : cdsv{ depth, stencil } {}
		VkClearDepthStencilValue cdsv;
	};

	struct ClearDepth {
		constexpr ClearDepth(float depth) noexcept : depth{ depth } {}
		float depth;

		operator ClearDepthStencil() {
			return ClearDepthStencil(depth, 0);
		}
	};

	struct ClearStencil {
		constexpr ClearStencil(uint32_t stencil) noexcept : stencil{ stencil } {}
		uint32_t stencil;
	};

	constexpr inline ClearDepthStencil operator|(ClearDepth d, ClearStencil s) noexcept {
		return ClearDepthStencil(d.depth, s.stencil);
	}

	static constexpr ClearDepth DepthOne = { 1.f };
	static constexpr ClearDepth DepthZero = { 0.f };

	static constexpr ClearDepthStencil DepthStencilOne = { 1.f, 1 };
	static constexpr ClearDepthStencil DepthStencilZero = { 0.f, 0 };

	struct Clear {
		constexpr Clear() = default;

		constexpr Clear(ClearColor cc) noexcept : is_color(true) {
			c.color = cc.ccv;
		}
		constexpr Clear(ClearDepth cc) noexcept : is_color(false) {
			c.depthStencil.depth = cc.depth;
		}
		constexpr Clear(ClearDepthStencil cc) noexcept : is_color(false) {
			c.depthStencil = cc.cdsv;
		}

		constexpr Clear(const Clear& other) noexcept {
			if (other.is_color) {
				c.color = other.c.color;
			} else {
				c.depthStencil = other.c.depthStencil;
			}
			is_color = other.is_color;
		}

		bool is_color;
		VkClearValue c;
	};

	enum Access : uint64_t {
		eNone = 1ULL << 0,       // as initial use: resource available without synchronization, as final use: resource does not need synchronizing
		eClear = 1ULL << 5,      // general clearing
		eColorRead = 1ULL << 7,  // read as a framebuffer color attachment
		eColorWrite = 1ULL << 8, // written as a framebuffer color attachment
		eColorRW = eColorRead | eColorWrite,
		eColorResolveRead = 1ULL << 10,  // special op to mark render pass resolve read
		eColorResolveWrite = 1ULL << 11, // special op to mark render pass resolve write
		eDepthStencilRead = 1ULL << 12,  // read as a framebuffer depth attachment
		eDepthStencilWrite = 1ULL << 13, // written as a framebuffer depth attachment
		eDepthStencilRW = eDepthStencilWrite | eDepthStencilRead,
		eVertexSampled = 1ULL << 15,   // sampled in a vertex shader
		eVertexRead = 1ULL << 16,      // read from an image or buffer in a vertex shader
		eAttributeRead = 1ULL << 17,   // read from an attribute in a vertex shader
		eIndexRead = 1ULL << 18,       // read from an index buffer for indexed rendering
		eIndirectRead = 1ULL << 19,    // read from an indirect buffer for indirect rendering
		eFragmentSampled = 1ULL << 20, // sampled in a fragment shader
		eFragmentRead = 1ULL << 21,    // read from an image or buffer in a fragment shader
		eFragmentWrite = 1ULL << 22,   // written using image store or buffer write in a fragment shader
		eFragmentRW = eFragmentRead | eFragmentWrite,
		eTransferRead = 1ULL << 23,  // read from an image or buffer in a transfer operation
		eTransferWrite = 1ULL << 24, // written from an image or buffer in a transfer operation
		eTransferRW = eTransferRead | eTransferWrite,
		eComputeRead = 1ULL << 25,  // read from an image or buffer in a compute shader
		eComputeWrite = 1ULL << 26, // written using image store or buffer write in a compute shader
		eComputeRW = eComputeRead | eComputeWrite,
		eComputeSampled = 1ULL << 27,  // sampled in a compute shader
		eRayTracingRead = 1ULL << 28,  // read from an image or buffer in a ray tracing shader
		eRayTracingWrite = 1ULL << 29, // written using image store or buffer write in a ray tracing shader
		eRayTracingRW = eRayTracingRead | eRayTracingWrite,
		eRayTracingSampled = 1ULL << 30,               // sampled in a ray tracing shader
		eAccelerationStructureBuildRead = 1ULL << 31,  // read during acceleration structure build
		eAccelerationStructureBuildWrite = 1ULL << 32, // written during acceleration structure build
		eAccelerationStructureBuildRW = eAccelerationStructureBuildRead | eAccelerationStructureBuildWrite,
		eHostRead = 1ULL << 33,  // read by the host
		eHostWrite = 1ULL << 34, // written by the host
		eHostRW = eHostRead | eHostWrite,
		eMemoryRead = 1ULL << 35,  // any device access that reads
		eMemoryWrite = 1ULL << 36, // any device access that writes
		eMemoryRW = eMemoryRead | eMemoryWrite,
		ePresent = 1ULL << 37 // presented from
	};

	inline constexpr Access operator|(Access bit0, Access bit1) noexcept {
		return Access((uint64_t)bit0 | (uint64_t)bit1);
	}

	enum class DomainFlagBits {
		eNone = 0,
		eHost = 1 << 0,
		ePE = 1 << 1,
		eGraphicsQueue = 1 << 2,
		eComputeQueue = 1 << 3,
		eTransferQueue = 1 << 4,
		eGraphicsOperation = 1 << 5,
		eComputeOperation = 1 << 6,
		eTransferOperation = 1 << 7,
		eDomainMask = 0b11111,
		eQueueMask = 0b11100,
		eOpMask = 0b11100000,
		eGraphicsOnGraphics = eGraphicsQueue | eGraphicsOperation,
		eComputeOnGraphics = eGraphicsQueue | eComputeOperation,
		eTransferOnGraphics = eGraphicsQueue | eTransferOperation,
		eComputeOnCompute = eComputeQueue | eComputeOperation,
		eTransferOnCompute = eComputeQueue | eComputeOperation,
		eTransferOnTransfer = eTransferQueue | eTransferOperation,
		eDevice = eGraphicsQueue | eComputeQueue | eTransferQueue,
		eAny = eDevice | eHost | ePE
	};

	using DomainFlags = Flags<DomainFlagBits>;
	inline constexpr DomainFlags operator|(DomainFlagBits bit0, DomainFlagBits bit1) noexcept {
		return DomainFlags(bit0) | bit1;
	}

	inline constexpr DomainFlags operator&(DomainFlagBits bit0, DomainFlagBits bit1) noexcept {
		return DomainFlags(bit0) & bit1;
	}

	inline constexpr DomainFlags operator^(DomainFlagBits bit0, DomainFlagBits bit1) noexcept {
		return DomainFlags(bit0) ^ bit1;
	}

	// Aligns given value up to nearest multiply of align value. For example: VmaAlignUp(11, 8) = 16.
	// Use types like uint32_t, uint64_t as T.
	// Source: VMA
	template<typename T>
	constexpr inline T align_up(T val, T align) noexcept {
		return (val + align - 1) / align * align;
	}

	struct ExecutorTag;

	struct ProfilingCallbacks {
		void* (*on_begin_command_buffer)(void* user_data, ExecutorTag tag, VkCommandBuffer cmdbuf) = nullptr;
		void (*on_end_command_buffer)(void* user_data, void* cbuf_data) = nullptr;

		void* (*on_begin_pass)(void* user_data, Name pass_name, class CommandBuffer& cmdbuf, DomainFlagBits domain) = nullptr;
		void (*on_end_pass)(void* user_data, void* pass_data) = nullptr;

		void* user_data = nullptr;
	};

	/// @brief Control compilation options when compiling the rendergraph
	struct RenderGraphCompileOptions {
		std::string graph_label;
		ProfilingCallbacks callbacks;
		bool dump_graph = false;
	};

	enum class DescriptorSetStrategyFlagBits {
		eDefault = 0, // implementation choice
		/* storage */
		ePerLayout = 1 << 1,      // one DS pool per layout
		eCommon = 1 << 2,         // common pool per layout
		ePushDescriptor = 1 << 3, // use push descriptors
	};

	using DescriptorSetStrategyFlags = Flags<DescriptorSetStrategyFlagBits>;
	inline constexpr DescriptorSetStrategyFlags operator|(DescriptorSetStrategyFlagBits bit0, DescriptorSetStrategyFlagBits bit1) noexcept {
		return DescriptorSetStrategyFlags(bit0) | bit1;
	}

	inline constexpr DescriptorSetStrategyFlags operator&(DescriptorSetStrategyFlagBits bit0, DescriptorSetStrategyFlagBits bit1) noexcept {
		return DescriptorSetStrategyFlags(bit0) & bit1;
	}

	inline constexpr DescriptorSetStrategyFlags operator^(DescriptorSetStrategyFlagBits bit0, DescriptorSetStrategyFlagBits bit1) noexcept {
		return DescriptorSetStrategyFlags(bit0) ^ bit1;
	}

	struct Node;
	struct Type;
	struct ChainLink;

	struct Ref {
		Node* node = nullptr;
		size_t index;

		std::shared_ptr<Type> type() const noexcept;
		ChainLink& link() noexcept;

		explicit constexpr operator bool() const noexcept {
			return node != nullptr;
		}

		constexpr std::strong_ordering operator<=>(const Ref&) const noexcept = default;
	};

	template<class Type, Access acc, class UniqueT>
	struct Arg {
		using type = Type;
		static constexpr Access access = acc;

		Type* ptr;

		Ref src;
		Ref def;

		operator const Type&() const noexcept
		  requires(!std::is_array_v<Type>)
		{
			return *ptr;
		}

		const Type* operator->() const noexcept
		  requires(!std::is_array_v<Type>)
		{
			return ptr;
		}

		size_t size() const noexcept
		  requires std::is_array_v<Type>;

		auto& operator[](size_t index) const noexcept
		  requires std::is_array_v<Type>
		{
			assert(index < size());
			return (*ptr)[index];
		}
	};
} // namespace vuk

#undef MOV
