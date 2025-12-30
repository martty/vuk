#pragma once

#include "vuk/runtime/vk/Image.hpp"
#include <array>
#include <span>

namespace vuk {
	struct AllocationEntry;
	struct ICI {
		ImageCreateFlags image_flags = {};
		ImageType image_type = ImageType::e2D;
		ImageTiling tiling = ImageTiling::eOptimal;
		ImageUsageFlags usage = {};
		Extent3D extent = {};
		Format format = Format::eUndefined;
		Samples sample_count = Samples::eInfer;
		bool allow_srgb_unorm_mutable = false;
		uint32_t level_count = VK_REMAINING_MIP_LEVELS;
		uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;

		operator VkImageCreateInfo() const noexcept {
			VkImageCreateInfo vkici{};
			vkici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
			vkici.pNext = nullptr;
			vkici.flags = (VkImageCreateFlags)image_flags;
			vkici.imageType = (VkImageType)image_type;
			vkici.format = (VkFormat)format;
			vkici.extent = (VkExtent3D)extent;
			vkici.mipLevels = level_count;
			vkici.arrayLayers = layer_count;
			vkici.samples = (VkSampleCountFlagBits)sample_count.count;
			vkici.tiling = (VkImageTiling)tiling;
			vkici.usage = (VkImageUsageFlags)usage;
			vkici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			vkici.queueFamilyIndexCount = 0;
			vkici.pQueueFamilyIndices = nullptr;
			vkici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			return vkici;
		}

		bool operator==(const ICI& o) const noexcept = default;
	};

	std::string format_as(const ICI& ici);

	struct ImageEntry : ICI {
		VkImage image;
		void* allocation;
		std::vector<uint32_t> image_view_indices;
	};
	struct ImageViewEntry;

	// Base classes for component access - only inherit from the one matching component_count
	template<typename T>
	struct ComponentAccessor1 {
		union {
			std::array<T, 1> data;
			struct {
				T r, x;
			};
		};
	};

	template<typename T>
	struct ComponentAccessor2 {
		union {
			std::array<T, 2> data;
			struct {
				T r, g;
			};
			struct {
				T x, y;
			};
		};
	};

	template<typename T>
	struct ComponentAccessor3 {
		union {
			std::array<T, 3> data;
			struct {
				T r, g, b;
			};
			struct {
				T x, y, z;
			};
		};
	};

	template<typename T>
	struct ComponentAccessor4 {
		union {
			std::array<T, 4> data;
			struct {
				T r, g, b, a;
			};
			struct {
				T x, y, z, w;
			};
		};
	};

	// Default base class for unknown/packed formats
	template<typename T, size_t Count>
	struct ComponentAccessorBase {
		std::array<T, (Count > 0 ? Count : 1)> data;
	};

	// Select the appropriate base class based on component count
	template<typename T, size_t Count, bool HasIndividual>
	struct SelectComponentAccessor {
		using type = ComponentAccessorBase<T, Count>;
	};

	template<typename T>
	struct SelectComponentAccessor<T, 1, true> {
		using type = ComponentAccessor1<T>;
	};

	template<typename T>
	struct SelectComponentAccessor<T, 2, true> {
		using type = ComponentAccessor2<T>;
	};

	template<typename T>
	struct SelectComponentAccessor<T, 3, true> {
		using type = ComponentAccessor3<T>;
	};

	template<typename T>
	struct SelectComponentAccessor<T, 4, true> {
		using type = ComponentAccessor4<T>;
	};

	// Map ComponentDataType to actual C++ types
	template<ComponentDataType CDT>
	struct data_type_to_cpp {
		using type = uint8_t; // default for eVoid/unknown
	};

	template<>
	struct data_type_to_cpp<ComponentDataType::eUint8> {
		using type = uint8_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eInt8> {
		using type = int8_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eUnorm8> {
		using type = uint8_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eSnorm8> {
		using type = int8_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eSrgb8> {
		using type = uint8_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eUint16> {
		using type = uint16_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eInt16> {
		using type = int16_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eUnorm16> {
		using type = uint16_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eSnorm16> {
		using type = int16_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eFloat16> {
		using type = uint16_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eUint32> {
		using type = uint32_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eInt32> {
		using type = int32_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eFloat32> {
		using type = float;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eUint64> {
		using type = uint64_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eInt64> {
		using type = int64_t;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::eFloat64> {
		using type = double;
	};
	template<>
	struct data_type_to_cpp<ComponentDataType::ePacked32> {
		using type = uint32_t;
	};

	template<Format format>
	struct FormatTraits {
		static constexpr ComponentDataType cdt = format_to_component_data_type(format);
		static constexpr size_t component_count = format_to_component_count(format);
		static constexpr bool has_individual_components = format_has_individual_components(format);
		using component_type = typename data_type_to_cpp<cdt>::type;
		static constexpr size_t size_bytes = sizeof(component_type) * component_count;
		using storage_type = std::array<component_type, (component_count > 0 ? component_count : 1)>;
	};

	// sRGB conversion helpers
	namespace detail {
		// Convert linear float [0,1] to sRGB uint8
		constexpr uint8_t linear_to_srgb8(float linear) noexcept {
			if (linear <= 0.0f)
				return 0;
			if (linear >= 1.0f)
				return 255;
			if (linear <= 0.0031308f)
				return static_cast<uint8_t>(linear * 12.92f * 255.0f + 0.5f);
			return static_cast<uint8_t>((1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f) * 255.0f + 0.5f);
		}

		// Convert sRGB uint8 to linear float [0,1]
		constexpr float srgb8_to_linear(uint8_t srgb) noexcept {
			float normalized = srgb / 255.0f;
			if (normalized <= 0.04045f)
				return normalized / 12.92f;
			return std::pow((normalized + 0.055f) / 1.055f, 2.4f);
		}
	} // namespace detail

	template<Format format = Format::eUndefined>
	struct ImageLike : std::integral_constant<Format, format> {
		static constexpr ComponentDataType cdt = format_to_component_data_type(format);
		using component_type = typename data_type_to_cpp<cdt>::type;
		static constexpr size_t component_count = format_to_component_count(format);
		static constexpr bool has_individual_components = format_has_individual_components(format);
		static constexpr size_t size_bytes = sizeof(component_type) * component_count;
		using storage_type = typename FormatTraits<format>::storage_type;

		storage_type data;

		// Constructors
		constexpr ImageLike() : data{} {}

		// Constructor for single component
		template<typename T = component_type>
		  requires(component_count == 1)
		constexpr ImageLike(T value) : data{ static_cast<component_type>(value) } {}

		// Constructor for 2 components
		template<typename T = component_type>
		  requires(component_count == 2)
		constexpr ImageLike(T r, T g) : data{ static_cast<component_type>(r), static_cast<component_type>(g) } {}

		// Constructor for 3 components
		template<typename T = component_type>
		  requires(component_count == 3)
		constexpr ImageLike(T r, T g, T b) : data{ static_cast<component_type>(r), static_cast<component_type>(g), static_cast<component_type>(b) } {}

		// Constructor for 4 components
		template<typename T = component_type>
		  requires(component_count == 4)
		constexpr ImageLike(T r, T g, T b, T a) :
		    data{ static_cast<component_type>(r), static_cast<component_type>(g), static_cast<component_type>(b), static_cast<component_type>(a) } {}

		// Constructor from packed 32-bit uint for 8-bit RGBA formats (ABGR layout in memory)
		constexpr ImageLike(uint32_t packed)
		  requires(component_count == 4 && sizeof(component_type) == 1)
		    :
		    data{ static_cast<component_type>(packed & 0xFF),
			        static_cast<component_type>((packed >> 8) & 0xFF),
			        static_cast<component_type>((packed >> 16) & 0xFF),
			        static_cast<component_type>((packed >> 24) & 0xFF) } {}

		// Conversion to packed 32-bit uint for 8-bit RGBA formats (ABGR layout in memory)
		constexpr uint32_t to_packed() const noexcept
		  requires(component_count == 4 && sizeof(component_type) == 1)
		{
			return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) | (static_cast<uint32_t>(data[2]) << 16) |
			       (static_cast<uint32_t>(data[3]) << 24);
		}

		// Array subscript operator - raw access
		constexpr component_type& operator[](size_t i) {
			return data[i];
		}

		constexpr const component_type& operator[](size_t i) const {
			return data[i];
		}

		// Component accessors with conversion for normalized formats
		// For UNORM8: stored as uint8, converted to/from float [0, 1]
		constexpr auto r() const noexcept
		  requires(component_count >= 1)
		{
			if constexpr (cdt == ComponentDataType::eUnorm8) {
				return data[0] / 255.0f;
			} else if constexpr (cdt == ComponentDataType::eSnorm8) {
				return std::max(data[0] / 127.0f, -1.0f);
			} else if constexpr (cdt == ComponentDataType::eSrgb8) {
				return detail::srgb8_to_linear(data[0]);
			} else if constexpr (cdt == ComponentDataType::eUnorm16) {
				return data[0] / 65535.0f;
			} else if constexpr (cdt == ComponentDataType::eSnorm16) {
				return std::max(data[0] / 32767.0f, -1.0f);
			} else {
				return data[0];
			}
		}

		constexpr void r(auto value) noexcept
		  requires(component_count >= 1)
		{
			if constexpr (cdt == ComponentDataType::eUnorm8) {
				data[0] = static_cast<uint8_t>(value * 255.0f + 0.5f);
			} else if constexpr (cdt == ComponentDataType::eSnorm8) {
				data[0] = static_cast<int8_t>(value * 127.0f);
			} else if constexpr (cdt == ComponentDataType::eSrgb8) {
				data[0] = detail::linear_to_srgb8(value);
			} else if constexpr (cdt == ComponentDataType::eUnorm16) {
				data[0] = static_cast<uint16_t>(value * 65535.0f + 0.5f);
			} else if constexpr (cdt == ComponentDataType::eSnorm16) {
				data[0] = static_cast<int16_t>(value * 32767.0f);
			} else {
				data[0] = static_cast<component_type>(value);
			}
		}

		constexpr auto g() const noexcept
		  requires(component_count >= 2)
		{
			if constexpr (cdt == ComponentDataType::eUnorm8) {
				return data[1] / 255.0f;
			} else if constexpr (cdt == ComponentDataType::eSnorm8) {
				return std::max(data[1] / 127.0f, -1.0f);
			} else if constexpr (cdt == ComponentDataType::eSrgb8) {
				return detail::srgb8_to_linear(data[1]);
			} else if constexpr (cdt == ComponentDataType::eUnorm16) {
				return data[1] / 65535.0f;
			} else if constexpr (cdt == ComponentDataType::eSnorm16) {
				return std::max(data[1] / 32767.0f, -1.0f);
			} else {
				return data[1];
			}
		}

		constexpr void g(auto value) noexcept
		  requires(component_count >= 2)
		{
			if constexpr (cdt == ComponentDataType::eUnorm8) {
				data[1] = static_cast<uint8_t>(value * 255.0f + 0.5f);
			} else if constexpr (cdt == ComponentDataType::eSnorm8) {
				data[1] = static_cast<int8_t>(value * 127.0f);
			} else if constexpr (cdt == ComponentDataType::eSrgb8) {
				data[1] = detail::linear_to_srgb8(value);
			} else if constexpr (cdt == ComponentDataType::eUnorm16) {
				data[1] = static_cast<uint16_t>(value * 65535.0f + 0.5f);
			} else if constexpr (cdt == ComponentDataType::eSnorm16) {
				data[1] = static_cast<int16_t>(value * 32767.0f);
			} else {
				data[1] = static_cast<component_type>(value);
			}
		}

		constexpr auto b() const noexcept
		  requires(component_count >= 3)
		{
			if constexpr (cdt == ComponentDataType::eUnorm8) {
				return data[2] / 255.0f;
			} else if constexpr (cdt == ComponentDataType::eSnorm8) {
				return std::max(data[2] / 127.0f, -1.0f);
			} else if constexpr (cdt == ComponentDataType::eSrgb8) {
				return detail::srgb8_to_linear(data[2]);
			} else if constexpr (cdt == ComponentDataType::eUnorm16) {
				return data[2] / 65535.0f;
			} else if constexpr (cdt == ComponentDataType::eSnorm16) {
				return std::max(data[2] / 32767.0f, -1.0f);
			} else {
				return data[2];
			}
		}

		constexpr void b(auto value) noexcept
		  requires(component_count >= 3)
		{
			if constexpr (cdt == ComponentDataType::eUnorm8) {
				data[2] = static_cast<uint8_t>(value * 255.0f + 0.5f);
			} else if constexpr (cdt == ComponentDataType::eSnorm8) {
				data[2] = static_cast<int8_t>(value * 127.0f);
			} else if constexpr (cdt == ComponentDataType::eSrgb8) {
				data[2] = detail::linear_to_srgb8(value);
			} else if constexpr (cdt == ComponentDataType::eUnorm16) {
				data[2] = static_cast<uint16_t>(value * 65535.0f + 0.5f);
			} else if constexpr (cdt == ComponentDataType::eSnorm16) {
				data[2] = static_cast<int16_t>(value * 32767.0f);
			} else {
				data[2] = static_cast<component_type>(value);
			}
		}

		constexpr auto a() const noexcept
		  requires(component_count >= 4)
		{
			if constexpr (cdt == ComponentDataType::eUnorm8) {
				return data[3] / 255.0f;
			} else if constexpr (cdt == ComponentDataType::eSnorm8) {
				return std::max(data[3] / 127.0f, -1.0f);
			} else if constexpr (cdt == ComponentDataType::eSrgb8) {
				// Alpha is always linear in sRGB formats
				return data[3] / 255.0f;
			} else if constexpr (cdt == ComponentDataType::eUnorm16) {
				return data[3] / 65535.0f;
			} else if constexpr (cdt == ComponentDataType::eSnorm16) {
				return std::max(data[3] / 32767.0f, -1.0f);
			} else {
				return data[3];
			}
		}

		constexpr void a(auto value) noexcept
		  requires(component_count >= 4)
		{
			if constexpr (cdt == ComponentDataType::eUnorm8) {
				data[3] = static_cast<uint8_t>(value * 255.0f + 0.5f);
			} else if constexpr (cdt == ComponentDataType::eSnorm8) {
				data[3] = static_cast<int8_t>(value * 127.0f);
			} else if constexpr (cdt == ComponentDataType::eSrgb8) {
				// Alpha is always linear in sRGB formats
				data[3] = static_cast<uint8_t>(value * 255.0f + 0.5f);
			} else if constexpr (cdt == ComponentDataType::eUnorm16) {
				data[3] = static_cast<uint16_t>(value * 65535.0f + 0.5f);
			} else if constexpr (cdt == ComponentDataType::eSnorm16) {
				data[3] = static_cast<int16_t>(value * 32767.0f);
			} else {
				data[3] = static_cast<component_type>(value);
			}
		}
	};

	struct Resolver {
		inline static thread_local Resolver* per_thread;

		Resolver();
		~Resolver();

		Resolver(const Resolver&) = delete;
		Resolver& operator=(const Resolver&) = delete;

		Resolver(Resolver&&) noexcept;
		Resolver& operator=(Resolver&&) noexcept;

		void commit(uint64_t base, size_t size, AllocationEntry ae);
		void decommit(uint64_t base, size_t size);

		uint64_t add_image(ImageEntry ve);
		void remove_image(uint64_t key);

		uint32_t add_image_view(ImageViewEntry ve);
		void remove_image_view(uint32_t key);

		struct BufferWithOffset {
			VkBuffer buffer;
			size_t offset;
		};

		struct BufferWithOffsetAndSize : BufferWithOffset {
			size_t size;
		};

		AllocationEntry& resolve_ptr(ptr_base ptr);
		BufferWithOffset ptr_to_buffer_offset(ptr_base ptr);
		ImageEntry& resolve_image(ptr_base ptr);
		ImageViewEntry& resolve_image_view(uint32_t view_key);

		void install_as_thread_resolver();
		void install_resolver_callbacks(VkDevice device, PFN_vkCreateImageView create_fn, PFN_vkDestroyImageView destroy_fn);

		// Debug/testing accessors
		size_t get_image_count() const;
		size_t get_active_image_view_count() const;

	private:
		struct ResolverImpl* impl;
	};

	template<class Type = byte>
	struct ptr : ptr_base {
		static constexpr bool imagelike = false;
		using pointed_T = Type;
		using UnwrappedT = detail::unwrap<Type>::T;

		UnwrappedT* operator->()
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			auto& ae = Resolver::per_thread->resolve_ptr(*this);
			auto offset = device_address - ae.buffer.base_address;
			return reinterpret_cast<UnwrappedT*>(ae.host_ptr + offset);
		}

		auto& operator*()
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			auto& ae = Resolver::per_thread->resolve_ptr(*this);
			auto offset = device_address - ae.buffer.base_address;
			return *reinterpret_cast<UnwrappedT*>(ae.host_ptr + offset);
		}

		const auto& operator*() const
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			auto& ae = Resolver::per_thread->resolve_ptr(*this);
			auto offset = device_address - ae.buffer.base_address;
			return *reinterpret_cast<const UnwrappedT*>(ae.host_ptr + offset);
		}

		auto& operator[](size_t index)
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			auto& ae = Resolver::per_thread->resolve_ptr(*this);
			auto offset = device_address - ae.buffer.base_address;
			return *(reinterpret_cast<std::remove_extent_t<UnwrappedT>*>(ae.host_ptr + offset) + index);
		}

		ptr operator+(size_t offset) const
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			return { device_address + offset * sizeof(UnwrappedT) };
		}

		void operator+=(size_t offset)
		  requires(!std::is_same_v<UnwrappedT, void>)
		{
			device_address += offset * sizeof(UnwrappedT);
		}
	};

	template<class T>
	using Unique_ptr = Unique<ptr<T>>;

	/// @brief Buffer creation parameters
	struct BufferCreateInfo {
		/// @brief Memory usage to determine which heap to allocate the memory from
		MemoryUsage memory_usage = MemoryUsage::eUnset;
		/// @brief Size of the allocation in bytes
		VkDeviceSize size = ~(0u);
		/// @brief Alignment of the allocation in bytes
		VkDeviceSize alignment = 1;

		std::strong_ordering operator<=>(const BufferCreateInfo&) const noexcept = default;
	};

	template<class T>
	struct member_type_helper;

	template<class C, class T>
	struct member_type_helper<T C::*> {
		using type = C;
	};

	template<class T>
	struct member_type : member_type_helper<typename std::remove_cvref<T>::type> {};

	// Helper type
	template<class T>
	using member_type_t = member_type<T>::type;

	template<auto MemberPtr>
	struct member_placeholder {
		/// @brief Returns the placeholder value for this member.
		static constexpr auto value = member_type_t<decltype(MemberPtr)>{}.*MemberPtr;
	};

	template<>
	struct member_placeholder<&BufferCreateInfo::alignment> {
		static constexpr VkDeviceSize value = 0;
	};

	inline std::string_view format_as(const MemoryUsage& foo) {
		switch (foo) {
		case MemoryUsage::eUnset:
			return "?";
		case MemoryUsage::eGPUonly:
			return "GPU";
		case MemoryUsage::eCPUonly:
			return "CPU";
		case MemoryUsage::eCPUtoGPU:
			return "C>G";
		case MemoryUsage::eGPUtoCPU:
			return "G>C";
		default:
			return "???";
		}
	}

	std::string format_as(const BufferCreateInfo& foo);

	/// @brief A contiguous portion of GPU-visible memory
	// fixed extent
	template<class Type, size_t Extent>
	struct view<BufferLike<Type>, Extent> {
		ptr<BufferLike<Type>> ptr;
		static constexpr size_t sz_bytes = Extent * sizeof(Type);

		view() = default;

		view(vuk::ptr<BufferLike<Type>> ptr)
		  requires(!std::is_array_v<Type>)
		    : ptr(ptr) {}

		auto& operator[](size_t index)
		  requires(!std::is_same_v<Type, void>)
		{
			assert(index < (sz_bytes / sizeof(Type)));
			return ptr[index];
		}

		const auto& operator[](size_t index) const
		  requires(!std::is_same_v<Type, void>)
		{
			assert(index < (sz_bytes / sizeof(Type)));
			return ptr[index];
		}

		explicit operator bool() const noexcept {
			return !!ptr;
		}

		[[nodiscard]] auto& data() noexcept {
			return ptr;
		}

		[[nodiscard]] size_t size_bytes() const noexcept {
			return sz_bytes;
		}

		[[nodiscard]] size_t count() const noexcept {
			return sz_bytes / sizeof(Type);
		}

		[[nodiscard]] view<BufferLike<byte>, sz_bytes> to_byte_view() const noexcept {
			return view<BufferLike<byte>, sz_bytes>{ vuk::ptr<BufferLike<byte>>{ ptr.device_address } };
		}

		template<class new_T>
		[[nodiscard]] view<BufferLike<new_T>, Extent * sizeof(Type) / sizeof(new_T)> cast() const noexcept {
			return view<BufferLike<new_T>, Extent * sizeof(Type) / sizeof(new_T)>{ vuk::ptr<BufferLike<new_T>>{ ptr.device_address } };
		}

		/// @brief Create a new view that is a subset of the original
		[[nodiscard]] view<BufferLike<Type>, dynamic_extent> subview(VkDeviceSize offset, VkDeviceSize new_count = ~(0ULL)) const {
			if (new_count == ~0ULL) {
				new_count = count() - offset;
			} else {
				assert(offset + new_count <= count());
			}
			return { ptr + offset, new_count };
		}

		[[nodiscard]] std::span<Type> to_span() noexcept {
			return std::span{ &*ptr, count() };
		}

		[[nodiscard]] std::span<const Type> to_span() const noexcept {
			return std::span{ &*ptr, count() };
		}

		operator view<BufferLike<Type>, dynamic_extent>() const noexcept {
			return { ptr, sz_bytes };
		}
		std::strong_ordering operator<=>(const view<BufferLike<Type>, Extent>&) const noexcept = default;
	};

	template<size_t FixedExtent>
	static view<BufferLike<Type>, FixedExtent> fixed_view(vuk::ptr<BufferLike<Type>> ptr) {
		return { ptr };
	}

	template<class Type>
	struct view<BufferLike<Type>, dynamic_extent> {
		ptr<BufferLike<Type>> ptr;
		size_t sz_bytes;

		view() = default;
		view(vuk::ptr<BufferLike<Type>> ptr, size_t count)
		  requires(!std::is_array_v<Type>)
		    : ptr(ptr), sz_bytes(count * sizeof(Type)) {}

		auto& operator[](size_t index)
		  requires(!std::is_same_v<Type, void>)
		{
			assert(index < (sz_bytes / sizeof(Type)));
			return ptr[index];
		}

		const auto& operator[](size_t index) const
		  requires(!std::is_same_v<Type, void>)
		{
			assert(index < (sz_bytes / sizeof(Type)));
			return ptr[index];
		}

		explicit operator bool() const noexcept {
			return !!ptr;
		}

		[[nodiscard]] auto& data() noexcept {
			return ptr;
		}

		[[nodiscard]] size_t size_bytes() const noexcept {
			return sz_bytes;
		}

		[[nodiscard]] size_t count() const noexcept {
			return sz_bytes / sizeof(Type);
		}

		[[nodiscard]] view<BufferLike<byte>, dynamic_extent> to_byte_view() const noexcept {
			return { vuk::ptr<BufferLike<byte>>{ ptr.device_address }, sz_bytes };
		}

		template<class new_T>
		[[nodiscard]] view<BufferLike<new_T>> cast() const noexcept {
			return { vuk::ptr<BufferLike<new_T>>{ ptr.device_address }, sz_bytes };
		}

		/// @brief Create a new view that is a subset of the original
		[[nodiscard]] view<BufferLike<Type>> subview(VkDeviceSize offset, VkDeviceSize new_count = ~(0ULL)) const {
			if (new_count == ~0ULL) {
				new_count = count() - offset;
			} else {
				assert(offset + new_count <= count());
			}
			return { ptr + offset, new_count };
		}

		template<class U = Type>
		[[nodiscard]] std::span<U> to_span() noexcept
		  requires(std::is_same_v<Type, byte> || std::is_same_v<Type, U>)
		{
			return std::span{ reinterpret_cast<U*>(&*ptr), count() };
		}

		template<class U = Type>
		[[nodiscard]] std::span<const U> to_span() const noexcept
		  requires(std::is_same_v<Type, byte> || std::is_same_v<Type, U>)
		{
			return std::span{ reinterpret_cast<const U*>(&*ptr), count() };
		}
		std::strong_ordering operator<=>(const view<BufferLike<Type>, dynamic_extent>&) const noexcept = default;
	};

	std::string format_as(const view<BufferLike<byte>, dynamic_extent>& foo);

	template<class Type>
	std::string format_as(const view<BufferLike<Type>, dynamic_extent>& foo) {
		return format_as(foo.to_byte_view());
	}

	template<class T, size_t Extent>
	struct is_view_type<view<T, Extent>> {
		static constexpr bool value = true;
	};

	template<class T>
	struct is_bufferlike_view_type<view<BufferLike<T>>> {
		static constexpr bool value = true;
	};

	template<class T, size_t Extent = dynamic_extent>
	using Unique_view = Unique<view<T, Extent>>;

	struct BufferViewCreateInfo {
		size_t elem_size;
		size_t count;
		Format format = Format::eUndefined;
	};

	/* struct ImageConstraints {
	  Format format = Format::eUndefined;
	  Samples samples = Samples::eInfer;
	};*/

	template<Format f>
	struct ptr<ImageLike<f>> : ptr_base {
		static constexpr bool imagelike = true;
		static constexpr Format static_format = f;

		using pointed_T = ImageLike<f>;
		using UnwrappedT = detail::unwrap<ImageLike<f>>::T;

		view<ImageLike<f>, dynamic_extent> default_view() {
			auto& ie = Resolver::per_thread->resolve_image(*this);
			return view<ImageLike<f>, dynamic_extent>{ ie.image_view_indices[0] };
		}
	};

	template<Format f = {}>
	using Image = ptr<ImageLike<f>>;

	template<Format f>
	struct create_info<Image<f>> {
		using type = ImageCreateInfo;
	};

	/*
	* 		Image<> image;
	  Format format = Format::eUndefined;
	  bool allow_srgb_unorm_mutable;
	  ImageViewCreateFlags image_view_flags;
	  ImageViewType view_type;
	  ComponentMapping components;
	  ImageLayout layout = ImageLayout::eUndefined;
	  ImageUsageFlags view_usage = {};

	  uint32_t base_level = VK_REMAINING_MIP_LEVELS;
	  uint32_t level_count = VK_REMAINING_MIP_LEVELS;

	  uint32_t base_layer = VK_REMAINING_ARRAY_LAYERS;
	  uint32_t layer_count = VK_REMAINING_ARRAY_LAYERS;*/

#pragma pack(push, 1)
	struct IVCI {
		uint32_t image_view_flags : 2;
		uint32_t allow_srgb_unorm_mutable : 1 = false;
		ImageViewType view_type : 3 = ImageViewType::e2D;
		ComponentSwizzle r_swizzle : 3 = ComponentSwizzle::eIdentity;
		ComponentSwizzle g_swizzle : 3 = ComponentSwizzle::eIdentity;
		ComponentSwizzle b_swizzle : 3 = ComponentSwizzle::eIdentity;
		ComponentSwizzle a_swizzle : 3 = ComponentSwizzle::eIdentity;
		uint32_t padding : 4 = 0;
		uint16_t base_level = 0xffff;
		uint16_t level_count = 0xffff;
		uint16_t base_layer = 0xffff;
		uint16_t layer_count = 0xffff;
		VkImageUsageFlags view_usage : 10;  // 8 bytes
		Image<> image = {};                 // 16 bytes
		Format format = Format::eUndefined; // 32 bytes in total

		static IVCI from(ImageViewCreateInfo ivci) {
			assert(ivci.pNext == nullptr && "Compression does not support pNextended IVCIs");
			IVCI to;
			to.image_view_flags = ivci.flags.m_mask;
			to.view_type = ivci.viewType;
			to.r_swizzle = ivci.components.r;
			to.g_swizzle = ivci.components.g;
			to.b_swizzle = ivci.components.b;
			to.a_swizzle = ivci.components.a;
			to.base_level = ivci.subresourceRange.baseMipLevel;
			to.level_count = ivci.subresourceRange.levelCount;
			to.base_layer = ivci.subresourceRange.baseArrayLayer;
			to.layer_count = ivci.subresourceRange.layerCount;
			to.format = ivci.format;
			to.view_usage = (VkImageUsageFlags)ivci.view_usage;
			return to;
		}

		explicit operator ImageViewCreateInfo() const noexcept {
			ImageViewCreateInfo ivci;
			ivci.flags = (ImageViewCreateFlags)image_view_flags;
			ivci.viewType = (ImageViewType)view_type;
			ivci.components.r = (ComponentSwizzle)r_swizzle;
			ivci.components.g = (ComponentSwizzle)g_swizzle;
			ivci.components.b = (ComponentSwizzle)b_swizzle;
			ivci.components.a = (ComponentSwizzle)a_swizzle;
			ivci.subresourceRange = { .aspectMask = (ImageAspectFlags)format_to_aspect(format),
				                        .baseMipLevel = base_level,
				                        .levelCount = level_count,
				                        .baseArrayLayer = base_layer,
				                        .layerCount = layer_count };
			ivci.image = Resolver::per_thread->resolve_image(image).image;
			ivci.format = (Format)format;
			ivci.view_usage = (ImageUsageFlags)view_usage;
			return ivci;
		}

		constexpr bool operator==(IVCI const& rhs) const noexcept = default;
	};
#pragma pack(pop)

	/// @brief Mip chain configuration for images
	enum class MipPreset : uint32_t {
		eNoMips = 0,         // No mip chain
		eFullMips = 1 << 16, // Full mip chain
	};

	constexpr inline MipPreset operator|(MipPreset a, MipPreset b) {
		return static_cast<MipPreset>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
	}

	constexpr inline MipPreset operator&(MipPreset a, MipPreset b) {
		return static_cast<MipPreset>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
	}

	constexpr inline bool operator!(MipPreset a) {
		return static_cast<uint32_t>(a) == 0;
	}

	/// @brief Usage flags for images
	enum class UsagePreset : uint32_t {
		eNone = 0,          // No usage
		eUpload = 1 << 0,   // Can be uploaded to
		eDownload = 1 << 1, // Can be downloaded from
		eCopy = 1 << 2,     // Can be used as copy source/destination
		eRender = 1 << 3,   // Can be rendered to (render target)
		eStore = 1 << 4,    // Can be stored to (storage image)
		eSampled = 1 << 5,  // Can be sampled from
	};

	constexpr inline UsagePreset operator|(UsagePreset a, UsagePreset b) {
		return static_cast<UsagePreset>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
	}

	constexpr inline UsagePreset operator&(UsagePreset a, UsagePreset b) {
		return static_cast<UsagePreset>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
	}

	constexpr inline UsagePreset& operator|=(UsagePreset& a, UsagePreset b) {
		a = a | b;
		return a;
	}

	constexpr inline bool operator!(UsagePreset a) {
		return static_cast<uint32_t>(a) == 0;
	}

	/// @brief Dimensionality preset for images
	enum class DimensionalityPreset : uint32_t {
		e2D = 0,        // 2D image
		e1D = 1 << 8,   // 1D image
		e3D = 2 << 8,   // 3D image
		eCube = 3 << 8, // Cube image
	};

	constexpr inline DimensionalityPreset operator|(DimensionalityPreset a, DimensionalityPreset b) {
		return static_cast<DimensionalityPreset>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
	}

	constexpr inline DimensionalityPreset operator&(DimensionalityPreset a, DimensionalityPreset b) {
		return static_cast<DimensionalityPreset>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
	}

	/// @brief Common image configuration presets combining usage and mip settings
	enum class Preset : uint32_t {
		eMap1D = static_cast<uint32_t>(UsagePreset::eUpload | UsagePreset::eSampled) | static_cast<uint32_t>(DimensionalityPreset::e1D) |
		         static_cast<uint32_t>(MipPreset::eFullMips), // 1D image with upload, sampled, never rendered to. Full mip chain. No arraying.
		eMap2D = static_cast<uint32_t>(UsagePreset::eUpload | UsagePreset::eSampled) | static_cast<uint32_t>(DimensionalityPreset::e2D) |
		         static_cast<uint32_t>(MipPreset::eFullMips), // 2D image with upload, sampled, never rendered to. Full mip chain. No arraying.
		eMap3D = static_cast<uint32_t>(UsagePreset::eUpload | UsagePreset::eSampled) | static_cast<uint32_t>(DimensionalityPreset::e3D) |
		         static_cast<uint32_t>(MipPreset::eFullMips), // 3D image with upload, sampled, never rendered to. Full mip chain. No arraying.
		eMapCube = static_cast<uint32_t>(UsagePreset::eUpload | UsagePreset::eSampled) | static_cast<uint32_t>(DimensionalityPreset::eCube) |
		           static_cast<uint32_t>(MipPreset::eFullMips), // Cubemap with upload, sampled, never rendered to. Full mip chain. No arraying.
		eRTT2D = static_cast<uint32_t>(UsagePreset::eSampled | UsagePreset::eRender) | static_cast<uint32_t>(DimensionalityPreset::e2D) |
		         static_cast<uint32_t>(MipPreset::eFullMips), // 2D image sampled and rendered to. Full mip chain. No arraying.
		eRTTCube = static_cast<uint32_t>(UsagePreset::eSampled | UsagePreset::eRender) | static_cast<uint32_t>(DimensionalityPreset::eCube) |
		           static_cast<uint32_t>(MipPreset::eFullMips), // Cubemap sampled and rendered to. Full mip chain. No arraying.
		eRTT2DUnmipped = static_cast<uint32_t>(UsagePreset::eSampled | UsagePreset::eRender) | static_cast<uint32_t>(DimensionalityPreset::e2D) |
		                 static_cast<uint32_t>(MipPreset::eNoMips), // 2D image sampled and rendered to. No mip chain. No arraying.
		eSTT2D = static_cast<uint32_t>(UsagePreset::eSampled | UsagePreset::eStore) | static_cast<uint32_t>(DimensionalityPreset::e2D) |
		         static_cast<uint32_t>(MipPreset::eFullMips), // 2D image sampled and stored to. Full mip chain. No arraying.
		eSTT2DUnmipped = static_cast<uint32_t>(UsagePreset::eSampled | UsagePreset::eStore) | static_cast<uint32_t>(DimensionalityPreset::e2D) |
		                 static_cast<uint32_t>(MipPreset::eNoMips), // 2D image sampled and stored to. No mip chain. No arraying.
		eGeneric2D = static_cast<uint32_t>(UsagePreset::eUpload | UsagePreset::eDownload | UsagePreset::eSampled | UsagePreset::eRender | UsagePreset::eStore) |
		             static_cast<uint32_t>(DimensionalityPreset::e2D) |
		             static_cast<uint32_t>(MipPreset::eFullMips), // 2D image with upload, download, sampling, rendering and storing. Full mip chain. No arraying.
	};

	// Utility functions for decomposing Preset
	inline MipPreset get_mip_preset(Preset preset) {
		return static_cast<MipPreset>(static_cast<uint32_t>(preset) & 0xFFFF0000);
	}

	inline UsagePreset get_usage_preset(Preset preset) {
		return static_cast<UsagePreset>(static_cast<uint32_t>(preset) & 0x00FF);
	}

	inline DimensionalityPreset get_dimensionality_preset(Preset preset) {
		return static_cast<DimensionalityPreset>(static_cast<uint32_t>(preset) & 0x0300);
	}

	inline Preset make_preset(UsagePreset usage, MipPreset mip, DimensionalityPreset dim = DimensionalityPreset::e2D) {
		return static_cast<Preset>(static_cast<uint32_t>(usage) | static_cast<uint32_t>(dim) | static_cast<uint32_t>(mip));
	}

	static ICI from_preset(Preset preset, Format format, Extent3D extent, Samples sample_count) {
		ICI ici = {};
		ici.format = format;
		ici.extent = extent;
		ici.sample_count = sample_count.count;

		UsagePreset usage_preset = get_usage_preset(preset);
		MipPreset mip_preset = get_mip_preset(preset);
		DimensionalityPreset dim_preset = get_dimensionality_preset(preset);

		// Set usage flags based on usage preset
		if (!!(usage_preset & UsagePreset::eUpload)) {
			ici.usage |= ImageUsageFlagBits::eTransferDst;
		}
		if (!!(usage_preset & UsagePreset::eDownload)) {
			ici.usage |= ImageUsageFlagBits::eTransferSrc;
		}
		if (!!(usage_preset & UsagePreset::eCopy)) {
			ici.usage |= ImageUsageFlagBits::eTransferSrc | ImageUsageFlagBits::eTransferDst;
		}
		if (!!(usage_preset & UsagePreset::eSampled)) {
			ici.usage |= ImageUsageFlagBits::eSampled;
		}
		if (!!(usage_preset & UsagePreset::eRender)) {
			ImageAspectFlags aspect = format_to_aspect(format);
			if (!!(aspect & ImageAspectFlagBits::eColor)) {
				ici.usage |= ImageUsageFlagBits::eColorAttachment;
			}
			if (!!(aspect & (ImageAspectFlagBits::eDepth | ImageAspectFlagBits::eStencil))) {
				ici.usage |= ImageUsageFlagBits::eDepthStencilAttachment;
			}
		}
		if (!!(usage_preset & UsagePreset::eStore)) {
			ici.usage |= ImageUsageFlagBits::eStorage;
		}

		// Set mip levels based on mip preset
		const uint32_t max_mips = (uint32_t)log2f((float)std::max(std::max(extent.width, extent.height), extent.depth)) + 1;
		if (!!(mip_preset & MipPreset::eFullMips)) {
			ici.level_count = max_mips;
		} else {
			ici.level_count = 1;
		}

		// Set image type and array layers based on dimensionality preset
		switch (dim_preset) {
		case DimensionalityPreset::e1D:
			ici.image_type = ImageType::e1D;
			ici.layer_count = 1;
			break;
		case DimensionalityPreset::e2D:
			ici.image_type = ImageType::e2D;
			ici.layer_count = 1;
			break;
		case DimensionalityPreset::e3D:
			ici.image_type = ImageType::e3D;
			ici.layer_count = 1;
			break;
		case DimensionalityPreset::eCube:
			ici.image_type = ImageType::e2D;
			ici.layer_count = 6;
			ici.image_flags = ImageCreateFlagBits::eCubeCompatible;
			break;
		default:
			assert(0);
		}

		return ici;
	}

	struct ImageViewEntry : IVCI {
		VkImageView api_view;
		size_t id;
		Extent3D extent;
		Samples sample_count;
		ImageLayout layout;
		size_t hash;
	};

	std::string format_as(const ImageViewEntry& entry);

	template<Format f>
	struct view<ImageLike<f>, dynamic_extent> {
		static constexpr auto static_format = f;

		uint32_t view_key = 0;

		view() = default;
		explicit view<ImageLike<f>, dynamic_extent>(uint32_t view_key) noexcept : view_key(view_key) {}

		explicit operator bool() const noexcept {
			return view_key != 0;
		}

		ImageViewEntry* operator->() const noexcept {
			return &Resolver::per_thread->resolve_image_view(view_key);
		}

		ImageViewEntry& get_meta() const noexcept {
			return Resolver::per_thread->resolve_image_view(view_key);
		}

		Format format() const noexcept {
			return get_meta().format;
		}

		view<ImageLike<f>, dynamic_extent> mip(uint32_t mip) const noexcept {
			auto a = get_meta();
			a.base_level = (a.base_level == VK_REMAINING_MIP_LEVELS ? 0 : a.base_level) + mip;
			a.level_count = 1;
			a.api_view = VK_NULL_HANDLE;
			return view<ImageLike<f>, dynamic_extent>{ Resolver::per_thread->add_image_view(a) };
		}

		view<ImageLike<f>, dynamic_extent> mip_range(uint32_t mip_base, uint32_t mip_count) const noexcept {
			auto a = get_meta();
			a.base_level = (a.base_level == VK_REMAINING_MIP_LEVELS ? 0 : a.base_level) + mip_base;
			a.level_count = mip_count;
			a.api_view = VK_NULL_HANDLE;
			return view<ImageLike<f>, dynamic_extent>{ Resolver::per_thread->add_image_view(a) };
		}

		view<ImageLike<f>, dynamic_extent> layer(uint32_t layer) const {
			auto a = get_meta();
			a.base_layer = (a.base_layer == VK_REMAINING_ARRAY_LAYERS ? 0 : a.base_layer) + layer;
			a.layer_count = 1;
			a.api_view = VK_NULL_HANDLE;
			return view<ImageLike<f>, dynamic_extent>{ Resolver::per_thread->add_image_view(a) };
		}

		view<ImageLike<f>, dynamic_extent> layer_range(uint32_t layer_base, uint32_t layer_count) const noexcept {
			auto a = get_meta();
			a.base_layer = (a.base_layer == VK_REMAINING_ARRAY_LAYERS ? 0 : a.base_layer) + layer_base;
			a.layer_count = layer_count;
			a.api_view = VK_NULL_HANDLE;
			return view<ImageLike<f>, dynamic_extent>{ Resolver::per_thread->add_image_view(a) };
		}

		Extent3D base_mip_extent() const noexcept {
			auto& ve = get_meta();
			auto& extent = ve.extent;
			auto base_level = ve.base_level;
			return { std::max(1u, extent.width >> base_level), std::max(1u, extent.height >> base_level), std::max(1u, extent.depth >> base_level) };
		}

		std::strong_ordering operator<=>(const view<ImageLike<f>, dynamic_extent>&) const noexcept = default;
	};

	template<Format f = {}>
	using ImageView = view<ImageLike<f>, dynamic_extent>;

	template<Format f>
	struct is_imagelike_view_type<view<ImageLike<f>>> {
		static constexpr bool value = true;
	};

	std::string format_as(const ImageView<Format::eUndefined>& foo);
	/*
	template<Format f>
	std::string format_as(const ImageView<f>& foo) {
	  return fmt::format("iv[{}]", foo.view_key);
	}*/

	template<Format f>
	struct create_info<ImageView<f>> {
		using type = IVCI;
	};

	template<Format f = {}>
	void synchronize(ImageView<f>, struct SyncHelper&);

	struct ImageWithIdentity {
		Image<> image;
	};

	struct CachedImageIdentifier {
		ICI ici;
		uint32_t id;
		uint32_t multi_frame_index;

		bool operator==(const CachedImageIdentifier&) const = default;
	};

	template<>
	struct create_info<ImageWithIdentity> {
		using type = CachedImageIdentifier;
	};

	struct AllocationEntry {
		byte* host_ptr;
		union {
			struct : BufferCreateInfo {
				VkBuffer buffer;
				size_t offset;
				uint64_t base_address;
			} buffer = {};
		};
		VkDeviceMemory device_memory;
		void* allocation;
		// enum class PTEFlags {} flags;
	};

	struct BVCI {
		ptr_base ptr;
		BufferViewCreateInfo vci;
	};
} // namespace vuk

namespace std {
	template<class T, size_t Extent>
	struct hash<vuk::view<vuk::BufferLike<T>, Extent>> {
		size_t operator()(vuk::view<vuk::BufferLike<T>, Extent> const& x) const {
			uint32_t v = std::hash<uint32_t>()(x.ptr.device_address);
			hash_combine_direct(v, std::hash<uint32_t>()(x.sz_bytes));
			return v;
		}
	};

	template<vuk::Format f>
	struct hash<vuk::ImageView<f>> {
		size_t operator()(vuk::ImageView<f> const& x) const noexcept {
			return std::hash<uint64_t>()(x.view_key);
		}
	};

	template<>
	struct hash<vuk::ImageViewEntry> {
		size_t operator()(vuk::ImageViewEntry const& x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.image.device_address, x.format, x.view_type, x.base_level, x.level_count, x.base_layer, x.layer_count, x.view_usage);
			return h;
		}
	};

	template<vuk::Format f>
	struct hash<vuk::Image<f>> {
		size_t operator()(vuk::Image<f> const& x) const noexcept {
			return std::hash<uint64_t>()(x.device_address);
		}
	};
} // namespace std