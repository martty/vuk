#pragma once

#include "vuk/Config.hpp"
#include "vuk/Hash.hpp"
#include "vuk/vuk_fwd.hpp"

#include <type_traits>

#define MOV(x) (static_cast<std::remove_reference_t<decltype(x)>&&>(x))

namespace vuk {
	struct HandleBase {
		size_t id = UINT64_MAX;
	};

	template<class T>
	struct Handle : public HandleBase {
		T payload;

		bool operator==(const Handle& o) const noexcept {
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

		Samples() : count(SampleCountFlagBits::e1) {}
		Samples(SampleCountFlagBits samples) : count(samples) {}
		Samples(Framebuffer) : count(SampleCountFlagBits::eInfer) {}

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

		bool operator==(Extent3D const& rhs) const noexcept {
			return (width == rhs.width) && (height == rhs.height) && (depth == rhs.depth);
		}

		bool operator!=(Extent3D const& rhs) const noexcept {
			return !operator==(rhs);
		}

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

	struct Dimension2D {
		Sizing sizing = Sizing::eAbsolute;

		Extent2D extent;

		struct {
			float width = 1.0f;
			float height = 1.0f;
		} _relative;

		static Dimension2D absolute(uint32_t width, uint32_t height) {
			return Dimension2D{ .extent = { width, height } };
		}
		static Dimension2D absolute(Extent2D extent) {
			return Dimension2D{ .extent = extent };
		}
		static Dimension2D relative(float width, float height) {
			return Dimension2D{ .sizing = Sizing::eRelative, ._relative = { .width = width, .height = height } };
		}
		static Dimension2D framebuffer() {
			return Dimension2D{ .sizing = Sizing::eRelative };
		}
	};

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

		static Rect2D absolute(int32_t x, int32_t y, uint32_t width, uint32_t height) {
			return Rect2D{ .offset = { x, y }, .extent = { width, height } };
		}
		static Rect2D absolute(Offset2D offset, Extent2D extent) {
			return Rect2D{ .offset = offset, .extent = extent };
		}
		static Rect2D relative(float x, float y, float width, float height) {
			return Rect2D{ .sizing = Sizing::eRelative, ._relative = { .x = x, .y = y, .width = width, .height = height } };
		}
		static Rect2D framebuffer() {
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
	uint32_t format_to_texel_block_size(vuk::Format) noexcept;
	// return the 3D texel block extent of a format
	Extent3D format_to_texel_block_extent(vuk::Format) noexcept;
	// compute the byte size of an image with given format and extent
	uint32_t compute_image_size(vuk::Format, vuk::Extent3D) noexcept;

	enum class IndexType {
		eUint16 = VK_INDEX_TYPE_UINT16,
		eUint32 = VK_INDEX_TYPE_UINT32,
		eNoneKHR = VK_INDEX_TYPE_NONE_KHR,
		eUint8EXT = VK_INDEX_TYPE_UINT8_EXT,
		eNoneNV = VK_INDEX_TYPE_NONE_NV
	};

	template<typename BitType>
	class Flags {
	public:
		using MaskType = typename std::underlying_type_t<BitType>;

		// constructors
		constexpr Flags() noexcept : m_mask(0) {}

		constexpr Flags(BitType bit) noexcept : m_mask(static_cast<MaskType>(bit)) {}

		constexpr Flags(Flags<BitType> const& rhs) noexcept : m_mask(rhs.m_mask) {}

		constexpr explicit Flags(MaskType flags) noexcept : m_mask(flags) {}

		constexpr bool operator<(Flags<BitType> const& rhs) const noexcept {
			return m_mask < rhs.m_mask;
		}

		constexpr bool operator<=(Flags<BitType> const& rhs) const noexcept {
			return m_mask <= rhs.m_mask;
		}

		constexpr bool operator>(Flags<BitType> const& rhs) const noexcept {
			return m_mask > rhs.m_mask;
		}

		constexpr bool operator>=(Flags<BitType> const& rhs) const noexcept {
			return m_mask >= rhs.m_mask;
		}

		constexpr bool operator==(Flags<BitType> const& rhs) const noexcept {
			return m_mask == rhs.m_mask;
		}

		constexpr bool operator!=(Flags<BitType> const& rhs) const noexcept {
			return m_mask != rhs.m_mask;
		}

		// logical operator
		constexpr bool operator!() const noexcept {
			return !m_mask;
		}

		// assignment operators
		constexpr Flags<BitType>& operator=(Flags<BitType> const& rhs) noexcept {
			m_mask = rhs.m_mask;
			return *this;
		}

		constexpr Flags<BitType>& operator|=(Flags<BitType> const& rhs) noexcept {
			m_mask |= rhs.m_mask;
			return *this;
		}

		constexpr Flags<BitType>& operator&=(Flags<BitType> const& rhs) noexcept {
			m_mask &= rhs.m_mask;
			return *this;
		}

		constexpr Flags<BitType>& operator^=(Flags<BitType> const& rhs) noexcept {
			m_mask ^= rhs.m_mask;
			return *this;
		}

		// cast operators
		explicit constexpr operator bool() const noexcept {
			return !!m_mask;
		}

		explicit constexpr operator MaskType() const noexcept {
			return m_mask;
		}

		// bitwise operators
		friend constexpr Flags<BitType> operator&(Flags<BitType> const& lhs, Flags<BitType> const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask & rhs.m_mask);
		}

		friend constexpr Flags<BitType> operator|(Flags<BitType> const& lhs, Flags<BitType> const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask | rhs.m_mask);
		}

		friend constexpr Flags<BitType> operator^(Flags<BitType> const& lhs, Flags<BitType> const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask ^ rhs.m_mask);
		}

		friend constexpr Flags<BitType> operator&(Flags<BitType> const& lhs, BitType const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask & (std::underlying_type_t<BitType>)rhs);
		}

		friend constexpr Flags<BitType> operator|(Flags<BitType> const& lhs, BitType const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask | (std::underlying_type_t<BitType>)rhs);
		}

		friend constexpr Flags<BitType> operator^(Flags<BitType> const& lhs, BitType const& rhs) noexcept {
			return Flags<BitType>(lhs.m_mask ^ (std::underlying_type_t<BitType>)rhs);
		}

		MaskType m_mask;
	};

	enum class ShaderStageFlagBits : VkShaderStageFlags {
		eVertex = VK_SHADER_STAGE_VERTEX_BIT,
		eTessellationControl = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
		eTessellationEvaluation = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
		eGeometry = VK_SHADER_STAGE_GEOMETRY_BIT,
		eFragment = VK_SHADER_STAGE_FRAGMENT_BIT,
		eCompute = VK_SHADER_STAGE_COMPUTE_BIT,
		eAllGraphics = VK_SHADER_STAGE_ALL_GRAPHICS,
		eAll = VK_SHADER_STAGE_ALL,
		eRaygenKHR = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
		eAnyHitKHR = VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
		eClosestHitKHR = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
		eMissKHR = VK_SHADER_STAGE_MISS_BIT_KHR,
		eIntersectionKHR = VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
		eCallableKHR = VK_SHADER_STAGE_CALLABLE_BIT_KHR,
		eTaskNV = VK_SHADER_STAGE_TASK_BIT_NV,
		eMeshNV = VK_SHADER_STAGE_MESH_BIT_NV,
		eAnyHitNV = VK_SHADER_STAGE_ANY_HIT_BIT_NV,
		eCallableNV = VK_SHADER_STAGE_CALLABLE_BIT_NV,
		eClosestHitNV = VK_SHADER_STAGE_CLOSEST_HIT_BIT_NV,
		eIntersectionNV = VK_SHADER_STAGE_INTERSECTION_BIT_NV,
		eMissNV = VK_SHADER_STAGE_MISS_BIT_NV,
		eRaygenNV = VK_SHADER_STAGE_RAYGEN_BIT_NV
	};

	using ShaderStageFlags = Flags<ShaderStageFlagBits>;
	inline constexpr ShaderStageFlags operator|(ShaderStageFlagBits bit0, ShaderStageFlagBits bit1) noexcept {
		return ShaderStageFlags(bit0) | bit1;
	}

	inline constexpr ShaderStageFlags operator&(ShaderStageFlagBits bit0, ShaderStageFlagBits bit1) noexcept {
		return ShaderStageFlags(bit0) & bit1;
	}

	inline constexpr ShaderStageFlags operator^(ShaderStageFlagBits bit0, ShaderStageFlagBits bit1) noexcept {
		return ShaderStageFlags(bit0) ^ bit1;
	}

	enum class PipelineStageFlagBits : VkPipelineStageFlags {
		eTopOfPipe = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		eDrawIndirect = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
		eVertexInput = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
		eVertexShader = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
		eTessellationControlShader = VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
		eTessellationEvaluationShader = VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
		eGeometryShader = VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
		eFragmentShader = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
		eEarlyFragmentTests = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
		eLateFragmentTests = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
		eColorAttachmentOutput = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		eComputeShader = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
		eTransfer = VK_PIPELINE_STAGE_TRANSFER_BIT,
		eBottomOfPipe = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
		eHost = VK_PIPELINE_STAGE_HOST_BIT,
		eAllGraphics = VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
		eAllCommands = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		eTransformFeedbackEXT = VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT,
		eConditionalRenderingEXT = VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT,
		eRayTracingShaderKHR = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
		eAccelerationStructureBuildKHR = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
		eShadingRateImageNV = VK_PIPELINE_STAGE_SHADING_RATE_IMAGE_BIT_NV,
		eTaskShaderNV = VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV,
		eMeshShaderNV = VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
		eFragmentDensityProcessEXT = VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT,
		eCommandPreprocessNV = VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV,
		eAccelerationStructureBuildNV = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_NV,
		eRayTracingShaderNV = VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_NV
	};

	using PipelineStageFlags = Flags<PipelineStageFlagBits>;
	inline constexpr PipelineStageFlags operator|(PipelineStageFlagBits bit0, PipelineStageFlagBits bit1) noexcept {
		return PipelineStageFlags(bit0) | bit1;
	}

	inline constexpr PipelineStageFlags operator&(PipelineStageFlagBits bit0, PipelineStageFlagBits bit1) noexcept {
		return PipelineStageFlags(bit0) & bit1;
	}

	inline constexpr PipelineStageFlags operator^(PipelineStageFlagBits bit0, PipelineStageFlagBits bit1) noexcept {
		return PipelineStageFlags(bit0) ^ bit1;
	}

	enum class AccessFlagBits : VkAccessFlags {
		eIndirectCommandRead = VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
		eIndexRead = VK_ACCESS_INDEX_READ_BIT,
		eVertexAttributeRead = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		eUniformRead = VK_ACCESS_UNIFORM_READ_BIT,
		eInputAttachmentRead = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,
		eShaderRead = VK_ACCESS_SHADER_READ_BIT,
		eShaderWrite = VK_ACCESS_SHADER_WRITE_BIT,
		eColorAttachmentRead = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
		eColorAttachmentWrite = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
		eDepthStencilAttachmentRead = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
		eDepthStencilAttachmentWrite = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
		eTransferRead = VK_ACCESS_TRANSFER_READ_BIT,
		eTransferWrite = VK_ACCESS_TRANSFER_WRITE_BIT,
		eHostRead = VK_ACCESS_HOST_READ_BIT,
		eHostWrite = VK_ACCESS_HOST_WRITE_BIT,
		eMemoryRead = VK_ACCESS_MEMORY_READ_BIT,
		eMemoryWrite = VK_ACCESS_MEMORY_WRITE_BIT,
		eTransformFeedbackWriteEXT = VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,
		eTransformFeedbackCounterReadEXT = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,
		eTransformFeedbackCounterWriteEXT = VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
		eConditionalRenderingReadEXT = VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT,
		eColorAttachmentReadNoncoherentEXT = VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,
		eAccelerationStructureReadKHR = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
		eAccelerationStructureWriteKHR = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
		eShadingRateImageReadNV = VK_ACCESS_SHADING_RATE_IMAGE_READ_BIT_NV,
		eFragmentDensityMapReadEXT = VK_ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,
		eCommandPreprocessReadNV = VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,
		eCommandPreprocessWriteNV = VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV,
		eAccelerationStructureReadNV = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_NV,
		eAccelerationStructureWriteNV = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_NV
	};

	using AccessFlags = Flags<AccessFlagBits>;

	inline constexpr AccessFlags operator|(AccessFlagBits bit0, AccessFlagBits bit1) noexcept {
		return AccessFlags(bit0) | bit1;
	}

	inline constexpr AccessFlags operator&(AccessFlagBits bit0, AccessFlagBits bit1) noexcept {
		return AccessFlags(bit0) & bit1;
	}

	inline constexpr AccessFlags operator^(AccessFlagBits bit0, AccessFlagBits bit1) noexcept {
		return AccessFlags(bit0) ^ bit1;
	}

	enum class MemoryUsage {
		eGPUonly = 1 /*VMA_MEMORY_USAGE_GPU_ONLY*/,
		eCPUtoGPU = 3 /*VMA_MEMORY_USAGE_CPU_TO_GPU*/,
		eCPUonly = 2 /*VMA_MEMORY_USAGE_CPU_ONLY*/,
		eGPUtoCPU = 4 /*VMA_MEMORY_USAGE_GPU_TO_CPU*/
	};

	using Bool32 = uint32_t;

	struct Preserve {};
	struct ClearColor {
		ClearColor(uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
			ccv.uint32[0] = r;
			ccv.uint32[1] = g;
			ccv.uint32[2] = b;
			ccv.uint32[3] = a;
		}
		ClearColor(float r, float g, float b, float a) {
			ccv.float32[0] = r;
			ccv.float32[1] = g;
			ccv.float32[2] = b;
			ccv.float32[3] = a;
		}
		VkClearColorValue ccv;
	};

	struct ClearDepthStencil {
		ClearDepthStencil(float depth, uint32_t stencil) {
			cdsv.depth = depth;
			cdsv.stencil = stencil;
		}
		VkClearDepthStencilValue cdsv;
	};

	struct PreserveOrClear {
		PreserveOrClear(ClearColor cc) : clear(true) {
			c.color = cc.ccv;
		}
		PreserveOrClear(ClearDepthStencil cc) : clear(true) {
			c.depthStencil = cc.cdsv;
		}
		PreserveOrClear(Preserve) : clear(false) {}

		bool clear;
		VkClearValue c;
	};

	struct Clear {
		Clear() = default;
		Clear(ClearColor cc) : is_color(true) {
			c.color = cc.ccv;
		}
		Clear(ClearDepthStencil cc) : is_color(false) {
			c.depthStencil = cc.cdsv;
		}

		Clear(const Clear& other) noexcept {
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

	enum Access {
		eNone,          // as initial use: resource available without synchronization, as final use: resource does not need synchronizing
		eInfer,         // as final use only: this use must be overwritten/inferred before compiling (internal)
		eConsume,       // must be overwritten before compiling: this access consumes this name (internal)
		eConverge,      // converge previous uses (internal)
		eManual,        // provided explictly (internal)
		eClear,         // general clearing
		eTransferClear, // vkCmdClearXXX
		eColorRW,
		eColorWrite,
		eColorRead,
		eColorResolveRead,  // special op to mark renderpass resolve read
		eColorResolveWrite, // special op to mark renderpass resolve write
		eDepthStencilRW,
		eDepthStencilRead,
		eInputRead,
		eVertexSampled,
		eVertexRead,
		eAttributeRead,
		eIndexRead,
		eIndirectRead,
		eFragmentSampled,
		eFragmentRead,
		eFragmentWrite, // written using image store
		eFragmentRW,
		eTransferRead,
		eTransferWrite,
		eComputeRead,
		eComputeWrite,
		eComputeRW,
		eComputeSampled,
		eHostRead,
		eHostWrite,
		eHostRW,
		eMemoryRead,
		eMemoryWrite,
		eMemoryRW,
		eRelease, // release a resource into a future (internal)
		eReleaseToGraphics,
		eReleaseToCompute,
		eReleaseToTransfer,
		eAcquire, // acquire a resource from a future (internal)
		eAcquireFromGraphics,
		eAcquireFromCompute,
		eAcquireFromTransfer
	};

	enum class DomainFlagBits {
		eNone = 0,
		eHost = 1 << 0,
		eGraphicsQueue = 1 << 1,
		eComputeQueue = 1 << 2,
		eTransferQueue = 1 << 3,
		eGraphicsOperation = 1 << 4,
		eComputeOperation = 1 << 5,
		eTransferOperation = 1 << 6,
		eQueueMask = 0b1110,
		eOpMask = 0b1110000,
		eGraphicsOnGraphics = eGraphicsQueue | eGraphicsOperation,
		eComputeOnGraphics = eGraphicsQueue | eComputeOperation,
		eTransferOnGraphics = eGraphicsQueue | eTransferOperation,
		eComputeOnCompute = eComputeQueue | eComputeOperation,
		eTransferOnCompute = eComputeQueue | eComputeOperation,
		eTransferOnTransfer = eTransferQueue | eTransferOperation,
		eDevice = eGraphicsQueue | eComputeQueue | eTransferQueue,
		eAny = eDevice | eHost
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
	static inline T align_up(T val, T align) {
		return (val + align - 1) / align * align;
	}

	struct CommandPool {
		VkCommandPool command_pool;
		uint32_t queue_family_index;

		constexpr bool operator==(const CommandPool& other) const noexcept {
			return command_pool == other.command_pool;
		}
	};

	struct CommandBufferAllocationCreateInfo {
		VkCommandBufferLevel level;
		CommandPool command_pool;
	};

	struct CommandBufferAllocation {
		CommandBufferAllocation() = default;
		CommandBufferAllocation(VkCommandBuffer command_buffer, CommandPool command_pool) : command_buffer(command_buffer), command_pool(command_pool) {}

		VkCommandBuffer command_buffer;
		CommandPool command_pool;

		operator VkCommandBuffer() {
			return command_buffer;
		}
	};
} // namespace vuk

namespace std {
	template<class BitType>
	struct hash<vuk::Flags<BitType>> {
		size_t operator()(vuk::Flags<BitType> const& x) const noexcept {
			return std::hash<typename vuk::Flags<BitType>::MaskType>()((typename vuk::Flags<BitType>::MaskType)x);
		}
	};
}; // namespace std

#undef MOV
