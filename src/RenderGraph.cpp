#include "vuk/RenderGraph.hpp"
#include "vuk/Hash.hpp" // for create
#include "Cache.hpp"
#include "vuk/Context.hpp"
#include "vuk/CommandBuffer.hpp"
#include "Allocator.hpp"
#include <unordered_set>

namespace vuk {
	template<class T, class A, class F>
	T* contains_if(std::vector<T, A>& v, F&& f) {
		auto it = std::find_if(v.begin(), v.end(), f);
		if (it != v.end()) return &(*it);
		else return nullptr;
	}

	template<class T, class A, class F>
	T const* contains_if(const std::vector<T, A>& v, F&& f) {
		auto it = std::find_if(v.begin(), v.end(), f);
		if (it != v.end()) return &(*it);
		else return nullptr;
	}

	template<class T, class A>
	T const* contains(const std::vector<T, A>& v, const T& f) {
		auto it = std::find(v.begin(), v.end(), f);
		if (it != v.end()) return &(*it);
		else return nullptr;
	}

	template <typename Iterator, typename Compare>
	void topological_sort(Iterator begin, Iterator end, Compare cmp) {
		while (begin != end) {
			auto const new_begin = std::partition(begin, end, [&](auto const& a) {
				return std::none_of(begin, end, [&](auto const& b) { return cmp(b, a); });
				});
			assert(new_begin != begin && "not a partial ordering");
			begin = new_begin;
		}
	}



	bool is_write_access(Access ia) {
		switch (ia) {
		case eColorResolveWrite:
		case eColorWrite:
		case eColorRW:
		case eDepthStencilRW:
		case eFragmentWrite:
		case eTransferDst:
		case eComputeWrite:
		case eComputeRW:
		case eHostWrite:
		case eHostRW:
		case eMemoryWrite:
		case eMemoryRW:
			return true;
		default:
			return false;
		}
	}

	bool is_read_access(Access ia) {
		switch (ia) {
		case eColorResolveRead:
		case eColorRead:
		case eColorRW:
		case eDepthStencilRead:
		case eDepthStencilRW:
		case eFragmentRead:
		case eFragmentSampled:
		case eTransferSrc:
		case eComputeRead:
		case eComputeSampled:
		case eComputeRW:
		case eHostRead:
		case eHostRW:
		case eMemoryRead:
		case eMemoryRW:
			return true;
		default:
			return false;
		}
	}

	Resource::Use to_use(Access ia) {
		switch (ia) {
		case eColorResolveWrite:
		case eColorWrite: return { vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentWrite, vuk::ImageLayout::eColorAttachmentOptimal };
		case eColorRW: return { vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentWrite | vuk::AccessFlagBits::eColorAttachmentRead, vuk::ImageLayout::eColorAttachmentOptimal };
		case eColorResolveRead:
		case eColorRead: return { vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentRead, vuk::ImageLayout::eColorAttachmentOptimal };
		case eDepthStencilRW: return { vuk::PipelineStageFlagBits::eEarlyFragmentTests | vuk::PipelineStageFlagBits::eLateFragmentTests, vuk::AccessFlagBits::eDepthStencilAttachmentRead | vuk::AccessFlagBits::eDepthStencilAttachmentWrite, vuk::ImageLayout::eDepthStencilAttachmentOptimal };

		case eFragmentSampled: return { vuk::PipelineStageFlagBits::eFragmentShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal };
		case eFragmentRead: return { vuk::PipelineStageFlagBits::eFragmentShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal };

		case eTransferSrc: return { vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferRead, vuk::ImageLayout::eTransferSrcOptimal };
		case eTransferDst: return { vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferWrite, vuk::ImageLayout::eTransferDstOptimal };

		case eComputeRead: return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eGeneral };
		case eComputeWrite: return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral };
		case eComputeRW: return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead | vuk::AccessFlagBits::eShaderWrite, vuk::ImageLayout::eGeneral };
		case eComputeSampled: return { vuk::PipelineStageFlagBits::eComputeShader, vuk::AccessFlagBits::eShaderRead, vuk::ImageLayout::eShaderReadOnlyOptimal };

		case eAttributeRead: return { vuk::PipelineStageFlagBits::eVertexInput, vuk::AccessFlagBits::eVertexAttributeRead, vuk::ImageLayout::eGeneral /* ignored */ };

		case eHostRead:
			return { vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostRead, vuk::ImageLayout::eGeneral };
		case eHostWrite:
			return { vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostWrite, vuk::ImageLayout::eGeneral };
		case eHostRW:
			return { vuk::PipelineStageFlagBits::eHost, vuk::AccessFlagBits::eHostRead | vuk::AccessFlagBits::eHostWrite, vuk::ImageLayout::eGeneral };

		case eMemoryRead: return { vuk::PipelineStageFlagBits::eBottomOfPipe, vuk::AccessFlagBits::eMemoryRead, vuk::ImageLayout::eGeneral };
		case eMemoryWrite: return { vuk::PipelineStageFlagBits::eBottomOfPipe, vuk::AccessFlagBits::eMemoryWrite, vuk::ImageLayout::eGeneral };
		case eMemoryRW: return { vuk::PipelineStageFlagBits::eBottomOfPipe, vuk::AccessFlagBits::eMemoryRead | vuk::AccessFlagBits::eMemoryWrite, vuk::ImageLayout::eGeneral };

		case eNone:
			return { vuk::PipelineStageFlagBits::eTopOfPipe, vuk::AccessFlagBits{}, vuk::ImageLayout::eUndefined };
		case eClear:
			return { vuk::PipelineStageFlagBits::eColorAttachmentOutput, vuk::AccessFlagBits::eColorAttachmentWrite, vuk::ImageLayout::ePreinitialized };
		case eTransferClear:
			return { vuk::PipelineStageFlagBits::eTransfer, vuk::AccessFlagBits::eTransferWrite, vuk::ImageLayout::eTransferDstOptimal };
		default:
			assert(0 && "NYI");
			return {};
		}

	}

	bool is_framebuffer_attachment(const Resource& r) {
		if (r.type == Resource::Type::eBuffer) return false;
		switch (r.ia) {
		case eColorWrite:
		case eColorRW:
		case eDepthStencilRW:
		case eColorRead:
		case eDepthStencilRead:
		case eColorResolveRead:
		case eColorResolveWrite:
			return true;
		default:
			return false;
		}
	}

	bool is_framebuffer_attachment(Resource::Use u) {
		switch (u.layout) {
		case vuk::ImageLayout::eColorAttachmentOptimal:
		case vuk::ImageLayout::eDepthStencilAttachmentOptimal:
			return true;
		default:
			return false;
		}
	}

	bool is_write_access(Resource::Use u) {
		if (u.access & vuk::AccessFlagBits::eColorAttachmentWrite) return true;
		if (u.access & vuk::AccessFlagBits::eDepthStencilAttachmentWrite) return true;
		if (u.access & vuk::AccessFlagBits::eShaderWrite) return true;
		if (u.access & vuk::AccessFlagBits::eTransferWrite) return true;
		if (u.access & vuk::AccessFlagBits::eHostWrite) return true;
		assert(0 && "NYI");
		return false;
	}

	bool is_read_access(Resource::Use u) {
		return !is_write_access(u);
	}

	vuk::ImageUsageFlags RenderGraph::compute_usage(const std::vector<vuk::RenderGraph::UseRef, short_alloc<UseRef, 64>>& chain) {
		vuk::ImageUsageFlags usage;
		for (const auto& c : chain) {
			switch (c.use.layout) {
			case vuk::ImageLayout::eDepthStencilAttachmentOptimal:
				usage |= vuk::ImageUsageFlagBits::eDepthStencilAttachment; break;
			case vuk::ImageLayout::eShaderReadOnlyOptimal: // TODO: more complex analysis
				usage |= vuk::ImageUsageFlagBits::eSampled; break;
			case vuk::ImageLayout::eColorAttachmentOptimal:
				usage |= vuk::ImageUsageFlagBits::eColorAttachment; break;
			case vuk::ImageLayout::eTransferSrcOptimal:
				usage |= vuk::ImageUsageFlagBits::eTransferSrc; break;
			case vuk::ImageLayout::eTransferDstOptimal:
				usage |= vuk::ImageUsageFlagBits::eTransferDst; break;
			default:;
			}
		}

		return usage;
	}

	uint32_t format_to_texel_block_size(vuk::Format format) noexcept {
		switch (format) {
		case vuk::Format::eR4G4UnormPack8:
		case vuk::Format::eR8Unorm:
		case vuk::Format::eR8Snorm:
		case vuk::Format::eR8Uscaled:
		case vuk::Format::eR8Sscaled:
		case vuk::Format::eR8Uint:
		case vuk::Format::eR8Sint:
		case vuk::Format::eR8Srgb:
			return 1;
		case vuk::Format::eR4G4B4A4UnormPack16:
		case vuk::Format::eB4G4R4A4UnormPack16:
		case vuk::Format::eR5G6B5UnormPack16:
		case vuk::Format::eB5G6R5UnormPack16:
		case vuk::Format::eR5G5B5A1UnormPack16:
		case vuk::Format::eB5G5R5A1UnormPack16:
		case vuk::Format::eA1R5G5B5UnormPack16:
		case vuk::Format::eR8G8Unorm:
		case vuk::Format::eR8G8Snorm:
		case vuk::Format::eR8G8Uscaled:
		case vuk::Format::eR8G8Sscaled:
		case vuk::Format::eR8G8Uint:
		case vuk::Format::eR8G8Sint:
		case vuk::Format::eR8G8Srgb:
		case vuk::Format::eR16Unorm:
		case vuk::Format::eR16Snorm:
		case vuk::Format::eR16Uscaled:
		case vuk::Format::eR16Sscaled:
		case vuk::Format::eR16Uint:
		case vuk::Format::eR16Sint:
		case vuk::Format::eR16Sfloat:
		case vuk::Format::eR10X6UnormPack16:
		case vuk::Format::eR12X4UnormPack16:
			return 2;
		case vuk::Format::eR8G8B8Unorm:
		case vuk::Format::eR8G8B8Snorm:
		case vuk::Format::eR8G8B8Uscaled:
		case vuk::Format::eR8G8B8Sscaled:
		case vuk::Format::eR8G8B8Uint:
		case vuk::Format::eR8G8B8Sint:
		case vuk::Format::eR8G8B8Srgb:
		case vuk::Format::eB8G8R8Unorm:
		case vuk::Format::eB8G8R8Snorm:
		case vuk::Format::eB8G8R8Uscaled:
		case vuk::Format::eB8G8R8Sscaled:
		case vuk::Format::eB8G8R8Uint:
		case vuk::Format::eB8G8R8Sint:
		case vuk::Format::eB8G8R8Srgb:
			return 3;
		case vuk::Format::eR8G8B8A8Unorm:
		case vuk::Format::eR8G8B8A8Snorm:
		case vuk::Format::eR8G8B8A8Uscaled:
		case vuk::Format::eR8G8B8A8Sscaled:
		case vuk::Format::eR8G8B8A8Uint:
		case vuk::Format::eR8G8B8A8Sint:
		case vuk::Format::eR8G8B8A8Srgb:
		case vuk::Format::eB8G8R8A8Unorm:
		case vuk::Format::eB8G8R8A8Snorm:
		case vuk::Format::eB8G8R8A8Uscaled:
		case vuk::Format::eB8G8R8A8Sscaled:
		case vuk::Format::eB8G8R8A8Uint:
		case vuk::Format::eB8G8R8A8Sint:
		case vuk::Format::eB8G8R8A8Srgb:
		case vuk::Format::eA8B8G8R8UnormPack32:
		case vuk::Format::eA8B8G8R8SnormPack32:
		case vuk::Format::eA8B8G8R8UscaledPack32:
		case vuk::Format::eA8B8G8R8SscaledPack32:
		case vuk::Format::eA8B8G8R8UintPack32:
		case vuk::Format::eA8B8G8R8SintPack32:
		case vuk::Format::eA8B8G8R8SrgbPack32:
		case vuk::Format::eA2R10G10B10UnormPack32:
		case vuk::Format::eA2R10G10B10SnormPack32:
		case vuk::Format::eA2R10G10B10UscaledPack32:
		case vuk::Format::eA2R10G10B10SscaledPack32:
		case vuk::Format::eA2R10G10B10UintPack32:
		case vuk::Format::eA2R10G10B10SintPack32:
		case vuk::Format::eA2B10G10R10UnormPack32:
		case vuk::Format::eA2B10G10R10SnormPack32:
		case vuk::Format::eA2B10G10R10UscaledPack32:
		case vuk::Format::eA2B10G10R10SscaledPack32:
		case vuk::Format::eA2B10G10R10UintPack32:
		case vuk::Format::eA2B10G10R10SintPack32:
		case vuk::Format::eR16G16Unorm:
		case vuk::Format::eR16G16Snorm:
		case vuk::Format::eR16G16Uscaled:
		case vuk::Format::eR16G16Sscaled:
		case vuk::Format::eR16G16Uint:
		case vuk::Format::eR16G16Sint:
		case vuk::Format::eR16G16Sfloat:
		case vuk::Format::eR32Uint:
		case vuk::Format::eR32Sint:
		case vuk::Format::eR32Sfloat:
		case vuk::Format::eB10G11R11UfloatPack32:
		case vuk::Format::eE5B9G9R9UfloatPack32:
		case vuk::Format::eR10X6G10X6Unorm2Pack16:
		case vuk::Format::eR12X4G12X4Unorm2Pack16:
			return 4;
		case vuk::Format::eG8B8G8R8422Unorm:
			return 4;
		case vuk::Format::eB8G8R8G8422Unorm:
			return 4;
		case vuk::Format::eR16G16B16Unorm:
		case vuk::Format::eR16G16B16Snorm:
		case vuk::Format::eR16G16B16Uscaled:
		case vuk::Format::eR16G16B16Sscaled:
		case vuk::Format::eR16G16B16Uint:
		case vuk::Format::eR16G16B16Sint:
		case vuk::Format::eR16G16B16Sfloat:
			return 6;
		case vuk::Format::eR16G16B16A16Unorm:
		case vuk::Format::eR16G16B16A16Snorm:
		case vuk::Format::eR16G16B16A16Uscaled:
		case vuk::Format::eR16G16B16A16Sscaled:
		case vuk::Format::eR16G16B16A16Uint:
		case vuk::Format::eR16G16B16A16Sint:
		case vuk::Format::eR16G16B16A16Sfloat:
		case vuk::Format::eR32G32Uint:
		case vuk::Format::eR32G32Sint:
		case vuk::Format::eR32G32Sfloat:
		case vuk::Format::eR64Uint:
		case vuk::Format::eR64Sint:
		case vuk::Format::eR64Sfloat:
			return 8;
		case vuk::Format::eR10X6G10X6B10X6A10X6Unorm4Pack16:
			return 8;
		case vuk::Format::eG10X6B10X6G10X6R10X6422Unorm4Pack16:
			return 8;
		case vuk::Format::eB10X6G10X6R10X6G10X6422Unorm4Pack16:
			return 8;
		case vuk::Format::eR12X4G12X4B12X4A12X4Unorm4Pack16:
			return 8;
		case vuk::Format::eG12X4B12X4G12X4R12X4422Unorm4Pack16:
			return 8;
		case vuk::Format::eB12X4G12X4R12X4G12X4422Unorm4Pack16:
			return 8;
		case vuk::Format::eG16B16G16R16422Unorm:
			return 8;
		case vuk::Format::eB16G16R16G16422Unorm:
			return 8;
		case vuk::Format::eR32G32B32Uint:
		case vuk::Format::eR32G32B32Sint:
		case vuk::Format::eR32G32B32Sfloat:
			return 12;
		case vuk::Format::eR32G32B32A32Uint:
		case vuk::Format::eR32G32B32A32Sint:
		case vuk::Format::eR32G32B32A32Sfloat:
		case vuk::Format::eR64G64Uint:
		case vuk::Format::eR64G64Sint:
		case vuk::Format::eR64G64Sfloat:
			return 16;
		case vuk::Format::eR64G64B64Uint:
		case vuk::Format::eR64G64B64Sint:
		case vuk::Format::eR64G64B64Sfloat:
			return 24;
		case vuk::Format::eR64G64B64A64Uint:
		case vuk::Format::eR64G64B64A64Sint:
		case vuk::Format::eR64G64B64A64Sfloat:
			return 32;
		case vuk::Format::eBc1RgbUnormBlock:
		case vuk::Format::eBc1RgbSrgbBlock:
			return 8;
		case vuk::Format::eBc1RgbaUnormBlock:
		case vuk::Format::eBc1RgbaSrgbBlock:
			return 8;
		case vuk::Format::eBc2UnormBlock:
		case vuk::Format::eBc2SrgbBlock:
			return 16;
		case vuk::Format::eBc3UnormBlock:
		case vuk::Format::eBc3SrgbBlock:
			return 16;
		case vuk::Format::eBc4UnormBlock:
		case vuk::Format::eBc4SnormBlock:
			return 8;
		case vuk::Format::eBc5UnormBlock:
		case vuk::Format::eBc5SnormBlock:
			return 16;
		case vuk::Format::eBc6HUfloatBlock:
		case vuk::Format::eBc6HSfloatBlock:
			return 16;
		case vuk::Format::eBc7UnormBlock:
		case vuk::Format::eBc7SrgbBlock:
			return 16;
		case vuk::Format::eEtc2R8G8B8UnormBlock:
		case vuk::Format::eEtc2R8G8B8SrgbBlock:
			return 8;
		case vuk::Format::eEtc2R8G8B8A1UnormBlock:
		case vuk::Format::eEtc2R8G8B8A1SrgbBlock:
			return 8;
		case vuk::Format::eEtc2R8G8B8A8UnormBlock:
		case vuk::Format::eEtc2R8G8B8A8SrgbBlock:
			return 8;
		case vuk::Format::eEacR11UnormBlock:
		case vuk::Format::eEacR11SnormBlock:
			return 8;
		case vuk::Format::eEacR11G11UnormBlock:
		case vuk::Format::eEacR11G11SnormBlock:
			return 16;
		case vuk::Format::eAstc4x4UnormBlock:
		case vuk::Format::eAstc4x4SfloatBlockEXT:
		case vuk::Format::eAstc4x4SrgbBlock:
			return 16;
		case vuk::Format::eAstc5x4UnormBlock:
		case vuk::Format::eAstc5x4SfloatBlockEXT:
		case vuk::Format::eAstc5x4SrgbBlock:
			return 16;
		case vuk::Format::eAstc5x5UnormBlock:
		case vuk::Format::eAstc5x5SfloatBlockEXT:
		case vuk::Format::eAstc5x5SrgbBlock:
			return 16;
		case vuk::Format::eAstc6x5UnormBlock:
		case vuk::Format::eAstc6x5SfloatBlockEXT:
		case vuk::Format::eAstc6x5SrgbBlock:
			return 16;
		case vuk::Format::eAstc6x6UnormBlock:
		case vuk::Format::eAstc6x6SfloatBlockEXT:
		case vuk::Format::eAstc6x6SrgbBlock:
			return 16;
		case vuk::Format::eAstc8x5UnormBlock:
		case vuk::Format::eAstc8x5SfloatBlockEXT:
		case vuk::Format::eAstc8x5SrgbBlock:
			return 16;
		case vuk::Format::eAstc8x6UnormBlock:
		case vuk::Format::eAstc8x6SfloatBlockEXT:
		case vuk::Format::eAstc8x6SrgbBlock:
			return 16;
		case vuk::Format::eAstc8x8UnormBlock:
		case vuk::Format::eAstc8x8SfloatBlockEXT:
		case vuk::Format::eAstc8x8SrgbBlock:
			return 16;
		case vuk::Format::eAstc10x5UnormBlock:
		case vuk::Format::eAstc10x5SfloatBlockEXT:
		case vuk::Format::eAstc10x5SrgbBlock:
			return 16;
		case vuk::Format::eAstc10x6UnormBlock:
		case vuk::Format::eAstc10x6SfloatBlockEXT:
		case vuk::Format::eAstc10x6SrgbBlock:
			return 16;
		case vuk::Format::eAstc10x8UnormBlock:
		case vuk::Format::eAstc10x8SfloatBlockEXT:
		case vuk::Format::eAstc10x8SrgbBlock:
			return 16;
		case vuk::Format::eAstc10x10UnormBlock:
		case vuk::Format::eAstc10x10SfloatBlockEXT:
		case vuk::Format::eAstc10x10SrgbBlock:
			return 16;
		case vuk::Format::eAstc12x10UnormBlock:
		case vuk::Format::eAstc12x10SfloatBlockEXT:
		case vuk::Format::eAstc12x10SrgbBlock:
			return 16;
		case vuk::Format::eAstc12x12UnormBlock:
		case vuk::Format::eAstc12x12SfloatBlockEXT:
		case vuk::Format::eAstc12x12SrgbBlock:
			return 16;
		case vuk::Format::eD16Unorm:
			return 2;
		case vuk::Format::eX8D24UnormPack32:
			return 4;
		case vuk::Format::eD32Sfloat:
			return 4;
		case vuk::Format::eS8Uint:
			return 1;
		case vuk::Format::eD16UnormS8Uint:
			return 3;
		case vuk::Format::eD24UnormS8Uint:
			return 4;
		case vuk::Format::eD32SfloatS8Uint:
			return 5;
		default:
			assert(0 && "format cannot be used with this function.");
			return 0;
		}
	}

	Extent3D format_to_texel_block_extent(vuk::Format format) noexcept {
		switch (format) {
		case vuk::Format::eR4G4UnormPack8:
		case vuk::Format::eR8Unorm:
		case vuk::Format::eR8Snorm:
		case vuk::Format::eR8Uscaled:
		case vuk::Format::eR8Sscaled:
		case vuk::Format::eR8Uint:
		case vuk::Format::eR8Sint:
		case vuk::Format::eR8Srgb:
		case vuk::Format::eR4G4B4A4UnormPack16:
		case vuk::Format::eB4G4R4A4UnormPack16:
		case vuk::Format::eR5G6B5UnormPack16:
		case vuk::Format::eB5G6R5UnormPack16:
		case vuk::Format::eR5G5B5A1UnormPack16:
		case vuk::Format::eB5G5R5A1UnormPack16:
		case vuk::Format::eA1R5G5B5UnormPack16:
		case vuk::Format::eR8G8Unorm:
		case vuk::Format::eR8G8Snorm:
		case vuk::Format::eR8G8Uscaled:
		case vuk::Format::eR8G8Sscaled:
		case vuk::Format::eR8G8Uint:
		case vuk::Format::eR8G8Sint:
		case vuk::Format::eR8G8Srgb:
		case vuk::Format::eR16Unorm:
		case vuk::Format::eR16Snorm:
		case vuk::Format::eR16Uscaled:
		case vuk::Format::eR16Sscaled:
		case vuk::Format::eR16Uint:
		case vuk::Format::eR16Sint:
		case vuk::Format::eR16Sfloat:
		case vuk::Format::eR10X6UnormPack16:
		case vuk::Format::eR12X4UnormPack16:
		case vuk::Format::eR8G8B8Unorm:
		case vuk::Format::eR8G8B8Snorm:
		case vuk::Format::eR8G8B8Uscaled:
		case vuk::Format::eR8G8B8Sscaled:
		case vuk::Format::eR8G8B8Uint:
		case vuk::Format::eR8G8B8Sint:
		case vuk::Format::eR8G8B8Srgb:
		case vuk::Format::eB8G8R8Unorm:
		case vuk::Format::eB8G8R8Snorm:
		case vuk::Format::eB8G8R8Uscaled:
		case vuk::Format::eB8G8R8Sscaled:
		case vuk::Format::eB8G8R8Uint:
		case vuk::Format::eB8G8R8Sint:
		case vuk::Format::eB8G8R8Srgb:
		case vuk::Format::eR8G8B8A8Unorm:
		case vuk::Format::eR8G8B8A8Snorm:
		case vuk::Format::eR8G8B8A8Uscaled:
		case vuk::Format::eR8G8B8A8Sscaled:
		case vuk::Format::eR8G8B8A8Uint:
		case vuk::Format::eR8G8B8A8Sint:
		case vuk::Format::eR8G8B8A8Srgb:
		case vuk::Format::eB8G8R8A8Unorm:
		case vuk::Format::eB8G8R8A8Snorm:
		case vuk::Format::eB8G8R8A8Uscaled:
		case vuk::Format::eB8G8R8A8Sscaled:
		case vuk::Format::eB8G8R8A8Uint:
		case vuk::Format::eB8G8R8A8Sint:
		case vuk::Format::eB8G8R8A8Srgb:
		case vuk::Format::eA8B8G8R8UnormPack32:
		case vuk::Format::eA8B8G8R8SnormPack32:
		case vuk::Format::eA8B8G8R8UscaledPack32:
		case vuk::Format::eA8B8G8R8SscaledPack32:
		case vuk::Format::eA8B8G8R8UintPack32:
		case vuk::Format::eA8B8G8R8SintPack32:
		case vuk::Format::eA8B8G8R8SrgbPack32:
		case vuk::Format::eA2R10G10B10UnormPack32:
		case vuk::Format::eA2R10G10B10SnormPack32:
		case vuk::Format::eA2R10G10B10UscaledPack32:
		case vuk::Format::eA2R10G10B10SscaledPack32:
		case vuk::Format::eA2R10G10B10UintPack32:
		case vuk::Format::eA2R10G10B10SintPack32:
		case vuk::Format::eA2B10G10R10UnormPack32:
		case vuk::Format::eA2B10G10R10SnormPack32:
		case vuk::Format::eA2B10G10R10UscaledPack32:
		case vuk::Format::eA2B10G10R10SscaledPack32:
		case vuk::Format::eA2B10G10R10UintPack32:
		case vuk::Format::eA2B10G10R10SintPack32:
		case vuk::Format::eR16G16Unorm:
		case vuk::Format::eR16G16Snorm:
		case vuk::Format::eR16G16Uscaled:
		case vuk::Format::eR16G16Sscaled:
		case vuk::Format::eR16G16Uint:
		case vuk::Format::eR16G16Sint:
		case vuk::Format::eR16G16Sfloat:
		case vuk::Format::eR32Uint:
		case vuk::Format::eR32Sint:
		case vuk::Format::eR32Sfloat:
		case vuk::Format::eB10G11R11UfloatPack32:
		case vuk::Format::eE5B9G9R9UfloatPack32:
		case vuk::Format::eR10X6G10X6Unorm2Pack16:
		case vuk::Format::eR12X4G12X4Unorm2Pack16:
		case vuk::Format::eG8B8G8R8422Unorm:
		case vuk::Format::eB8G8R8G8422Unorm:
		case vuk::Format::eR16G16B16Unorm:
		case vuk::Format::eR16G16B16Snorm:
		case vuk::Format::eR16G16B16Uscaled:
		case vuk::Format::eR16G16B16Sscaled:
		case vuk::Format::eR16G16B16Uint:
		case vuk::Format::eR16G16B16Sint:
		case vuk::Format::eR16G16B16Sfloat:
		case vuk::Format::eR16G16B16A16Unorm:
		case vuk::Format::eR16G16B16A16Snorm:
		case vuk::Format::eR16G16B16A16Uscaled:
		case vuk::Format::eR16G16B16A16Sscaled:
		case vuk::Format::eR16G16B16A16Uint:
		case vuk::Format::eR16G16B16A16Sint:
		case vuk::Format::eR16G16B16A16Sfloat:
		case vuk::Format::eR32G32Uint:
		case vuk::Format::eR32G32Sint:
		case vuk::Format::eR32G32Sfloat:
		case vuk::Format::eR64Uint:
		case vuk::Format::eR64Sint:
		case vuk::Format::eR64Sfloat:
		case vuk::Format::eR10X6G10X6B10X6A10X6Unorm4Pack16:
		case vuk::Format::eG10X6B10X6G10X6R10X6422Unorm4Pack16:
		case vuk::Format::eB10X6G10X6R10X6G10X6422Unorm4Pack16:
		case vuk::Format::eR12X4G12X4B12X4A12X4Unorm4Pack16:
		case vuk::Format::eG12X4B12X4G12X4R12X4422Unorm4Pack16:
		case vuk::Format::eB12X4G12X4R12X4G12X4422Unorm4Pack16:
		case vuk::Format::eG16B16G16R16422Unorm:
		case vuk::Format::eB16G16R16G16422Unorm:
		case vuk::Format::eR32G32B32Uint:
		case vuk::Format::eR32G32B32Sint:
		case vuk::Format::eR32G32B32Sfloat:
		case vuk::Format::eR32G32B32A32Uint:
		case vuk::Format::eR32G32B32A32Sint:
		case vuk::Format::eR32G32B32A32Sfloat:
		case vuk::Format::eR64G64Uint:
		case vuk::Format::eR64G64Sint:
		case vuk::Format::eR64G64Sfloat:
		case vuk::Format::eR64G64B64Uint:
		case vuk::Format::eR64G64B64Sint:
		case vuk::Format::eR64G64B64Sfloat:
		case vuk::Format::eR64G64B64A64Uint:
		case vuk::Format::eR64G64B64A64Sint:
		case vuk::Format::eR64G64B64A64Sfloat:
			return { 1,1,1 };
		case vuk::Format::eBc1RgbUnormBlock:
		case vuk::Format::eBc1RgbSrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eBc1RgbaUnormBlock:
		case vuk::Format::eBc1RgbaSrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eBc2UnormBlock:
		case vuk::Format::eBc2SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eBc3UnormBlock:
		case vuk::Format::eBc3SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eBc4UnormBlock:
		case vuk::Format::eBc4SnormBlock:
			return { 4,4,1 };
		case vuk::Format::eBc5UnormBlock:
		case vuk::Format::eBc5SnormBlock:
			return{ 4,4,1 };
		case vuk::Format::eBc6HUfloatBlock:
		case vuk::Format::eBc6HSfloatBlock:
			return{ 4,4,1 };
		case vuk::Format::eBc7UnormBlock:
		case vuk::Format::eBc7SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eEtc2R8G8B8UnormBlock:
		case vuk::Format::eEtc2R8G8B8SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eEtc2R8G8B8A1UnormBlock:
		case vuk::Format::eEtc2R8G8B8A1SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eEtc2R8G8B8A8UnormBlock:
		case vuk::Format::eEtc2R8G8B8A8SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eEacR11UnormBlock:
		case vuk::Format::eEacR11SnormBlock:
			return { 4,4,1 };
		case vuk::Format::eEacR11G11UnormBlock:
		case vuk::Format::eEacR11G11SnormBlock:
			return { 4,4,1 };
		case vuk::Format::eAstc4x4UnormBlock:
		case vuk::Format::eAstc4x4SfloatBlockEXT:
		case vuk::Format::eAstc4x4SrgbBlock:
			return { 4,4,1 };
		case vuk::Format::eAstc5x4UnormBlock:
		case vuk::Format::eAstc5x4SfloatBlockEXT:
		case vuk::Format::eAstc5x4SrgbBlock:
			return { 5,4,1 };
		case vuk::Format::eAstc5x5UnormBlock:
		case vuk::Format::eAstc5x5SfloatBlockEXT:
		case vuk::Format::eAstc5x5SrgbBlock:
			return { 5,5,1 };
		case vuk::Format::eAstc6x5UnormBlock:
		case vuk::Format::eAstc6x5SfloatBlockEXT:
		case vuk::Format::eAstc6x5SrgbBlock:
			return { 6,5,1 };
		case vuk::Format::eAstc6x6UnormBlock:
		case vuk::Format::eAstc6x6SfloatBlockEXT:
		case vuk::Format::eAstc6x6SrgbBlock:
			return { 6,6,1 };
		case vuk::Format::eAstc8x5UnormBlock:
		case vuk::Format::eAstc8x5SfloatBlockEXT:
		case vuk::Format::eAstc8x5SrgbBlock:
			return { 8,5,1 };
		case vuk::Format::eAstc8x6UnormBlock:
		case vuk::Format::eAstc8x6SfloatBlockEXT:
		case vuk::Format::eAstc8x6SrgbBlock:
			return { 8,6,1 };
		case vuk::Format::eAstc8x8UnormBlock:
		case vuk::Format::eAstc8x8SfloatBlockEXT:
		case vuk::Format::eAstc8x8SrgbBlock:
			return { 8,8,1 };
		case vuk::Format::eAstc10x5UnormBlock:
		case vuk::Format::eAstc10x5SfloatBlockEXT:
		case vuk::Format::eAstc10x5SrgbBlock:
			return { 10, 5, 1 };
		case vuk::Format::eAstc10x6UnormBlock:
		case vuk::Format::eAstc10x6SfloatBlockEXT:
		case vuk::Format::eAstc10x6SrgbBlock:
			return {10, 6, 1};
		case vuk::Format::eAstc10x8UnormBlock:
		case vuk::Format::eAstc10x8SfloatBlockEXT:
		case vuk::Format::eAstc10x8SrgbBlock:
			return {10, 8, 1};
		case vuk::Format::eAstc10x10UnormBlock:
		case vuk::Format::eAstc10x10SfloatBlockEXT:
		case vuk::Format::eAstc10x10SrgbBlock:
			return {10, 10, 1};
		case vuk::Format::eAstc12x10UnormBlock:
		case vuk::Format::eAstc12x10SfloatBlockEXT:
		case vuk::Format::eAstc12x10SrgbBlock:
			return {12, 10, 1};
		case vuk::Format::eAstc12x12UnormBlock:
		case vuk::Format::eAstc12x12SfloatBlockEXT:
		case vuk::Format::eAstc12x12SrgbBlock:
			return {12, 12, 1};
		case vuk::Format::eD16Unorm:
		case vuk::Format::eX8D24UnormPack32:
		case vuk::Format::eD32Sfloat:
		case vuk::Format::eS8Uint:
		case vuk::Format::eD16UnormS8Uint:
		case vuk::Format::eD24UnormS8Uint:
		case vuk::Format::eD32SfloatS8Uint:
			return { 1,1,1 };
		default:
			assert(0 && "format cannot be used with this function.");
			return { 0,0,0 };
		}
	}

	uint32_t compute_image_size(vuk::Format format, vuk::Extent3D extent) noexcept {
		auto block_extent = format_to_texel_block_extent(format);
		auto extent_in_blocks = Extent3D{
			(extent.width + block_extent.width - 1) / block_extent.width,
			(extent.height + block_extent.height - 1) / block_extent.height,
			(extent.depth + block_extent.depth - 1) / block_extent.depth
		};
		return extent_in_blocks.width * extent_in_blocks.height * extent_in_blocks.depth * format_to_texel_block_size(format);
	}

	uint32_t Ignore::to_size() {
		if (bytes != 0) return bytes;
		return format_to_texel_block_size(format);
	}

	FormatOrIgnore::FormatOrIgnore(vuk::Format format) : ignore(false), format(format), size(format_to_texel_block_size(format)) {
	}
	FormatOrIgnore::FormatOrIgnore(Ignore ign) : ignore(true), format(ign.format), size(ign.to_size()) {
	}

#define INIT(x) x(decltype(x)::allocator_type(*arena_))
#define INIT2(x) x(decltype(x)::allocator_type(arena_))

	RenderGraph::RenderGraph() : arena_(new arena(1024 * 128)), INIT(head_passes), INIT(tail_passes), INIT(global_io), INIT(rpis) {
		passes.reserve(64);
	}

	// determine rendergraph inputs and outputs, and resources that are neither
	void RenderGraph::build_io() {
		std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 8>> inputs{ *arena_ };
		std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 8>> outputs{ *arena_ };

		for (auto& pif : passes) {
			for (auto& res : pif.pass.resources) {
				if (is_read_access(res.ia)) {
					pif.inputs.insert(res);
				}
				if (is_write_access(res.ia)) {
					pif.outputs.insert(res);
				}
			}

			for (auto& i : pif.inputs) {
				if (global_outputs.erase(i) == 0) {
					pif.global_inputs.insert(i);
				}
			}
			for (auto& i : pif.outputs) {
				if (global_inputs.erase(i) == 0) {
					pif.global_outputs.insert(i);
				}
			}

			global_inputs.insert(pif.global_inputs.begin(), pif.global_inputs.end());
			global_outputs.insert(pif.global_outputs.begin(), pif.global_outputs.end());

			inputs.insert(pif.inputs.begin(), pif.inputs.end());
			outputs.insert(pif.outputs.begin(), pif.outputs.end());
		}

		global_io.insert(global_io.end(), global_inputs.begin(), global_inputs.end());
		global_io.insert(global_io.end(), global_outputs.begin(), global_outputs.end());
		global_io.erase(std::unique(global_io.begin(), global_io.end()), global_io.end());
	}

	template<class T>
	Name resolve_name(Name in, const T& aliases) {
		auto it = aliases.find(in);
		if (it == aliases.end())
			return in;
		else
			return resolve_name(it->second, aliases);
	};

	void RenderGraph::build() {
		// compute sync
		// find which reads are graph inputs (not produced by any pass) & outputs (not consumed by any pass)
		build_io();
		// sort passes
		if (passes.size() > 1) {
			topological_sort(passes.begin(), passes.end(), [](const auto& p1, const auto& p2) {
				bool could_execute_after = false;
				bool could_execute_before = false;
				for (auto& o : p1.outputs) {
					for (auto& i : p2.inputs) {
						if (i.hash_src_name == o.hash_use_name)
							could_execute_after = true;
					}
				}
				for (auto& o : p2.outputs) {
					for (auto& i : p1.inputs) {
						if (i.hash_src_name == o.hash_use_name)
							could_execute_before = true;
					}
				}
				if (!could_execute_after && !could_execute_before && p1.outputs == p2.outputs) {
					return p1.pass.auxiliary_order < p2.pass.auxiliary_order;
				}

				if (could_execute_after && could_execute_before) {
					return p1.pass.auxiliary_order < p2.pass.auxiliary_order;
				} else if (could_execute_after) {
					return true;
				} else
					return false;
				});
		}
		// determine which passes are "head" and "tail" -> ones that can execute in the beginning or the end of the RG
		// -> the ones that only have global io
		for (auto& p : passes) {
			if (p.global_inputs.size() == p.inputs.size()) {
				head_passes.push_back(&p);
				p.is_head_pass = true;
			}
			if (p.global_outputs.size() == p.outputs.size()) {
				tail_passes.push_back(&p);
				p.is_tail_pass = true;
			}
		}

		// assemble use chains
		for (auto& passinfo : passes) {
			for (auto& res : passinfo.pass.resources) {
				if (res.src_name != res.use_name) {
					aliases[res.use_name] = res.src_name;
				}
				auto it = use_chains.find(resolve_name(res.use_name, aliases));
				if (it == use_chains.end()) {
					it = use_chains.emplace(resolve_name(res.use_name, aliases), std::vector<UseRef, short_alloc<UseRef, 64>>{short_alloc<UseRef, 64>{*arena_}}).first;
				}
				it->second.emplace_back(UseRef{ to_use(res.ia), &passinfo });
			}
		}

		// we need to collect passes into framebuffers, which will determine the renderpasses
		using attachment_set = std::unordered_set<Resource, std::hash<Resource>, std::equal_to<Resource>, short_alloc<Resource, 16>>;
		using passinfo_vec = std::vector<PassInfo*, short_alloc<PassInfo*, 16>>;
		std::vector<std::pair<attachment_set, passinfo_vec>, short_alloc<std::pair<attachment_set, passinfo_vec>, 8>> attachment_sets{ *arena_ };
		for (auto& passinfo : passes) {
			attachment_set atts{ *arena_ };

			for (auto& res : passinfo.pass.resources) {
				if (is_framebuffer_attachment(res))
					atts.insert(res);
			}

			if (auto p = attachment_sets.size() > 0 && attachment_sets.back().first == atts ? &attachment_sets.back() : nullptr) {
				p->second.push_back(&passinfo);
			} else {
				passinfo_vec pv{ *arena_ };
				pv.push_back(&passinfo);
				attachment_sets.emplace_back(atts, pv);
			}
		}

		// renderpasses are uniquely identified by their index from now on
		// tell passes in which renderpass/subpass they will execute
		rpis.reserve(attachment_sets.size());
		for (auto& [attachments, passes] : attachment_sets) {
			RenderPassInfo rpi{ *arena_ };
			auto rpi_index = rpis.size();

			int32_t subpass = -1;
			for (auto& p : passes) {
				p->render_pass_index = rpi_index;
				if (rpi.subpasses.size() > 0) {
					auto& last_pass = rpi.subpasses.back().passes[0];
					// if the pass has the same inputs and outputs, we execute them on the same subpass
					if (last_pass->inputs == p->inputs && last_pass->outputs == p->outputs) {
						p->subpass = last_pass->subpass;
						rpi.subpasses.back().passes.push_back(p);
						// potentially upgrade to secondary cbufs
						rpi.subpasses.back().use_secondary_command_buffers |= p->pass.use_secondary_command_buffers;
						continue;
					}
				}
				SubpassInfo si{ *arena_ };
				si.passes = { p };
				si.use_secondary_command_buffers = p->pass.use_secondary_command_buffers;
				p->subpass = ++subpass;
				rpi.subpasses.push_back(si);
			}
			for (auto& att : attachments) {
				AttachmentRPInfo info;
				info.name = resolve_name(att.use_name, aliases);
				rpi.attachments.push_back(info);
			}

			if (attachments.size() == 0) {
				rpi.framebufferless = true;
			}

			rpis.push_back(rpi);
		}
	}

	void RenderGraph::bind_attachment_to_swapchain(Name name, SwapchainRef swp, Clear c) {
		AttachmentRPInfo attachment_info;
		attachment_info.extents = swp->extent;
		attachment_info.iv = {};
		// directly presented
		attachment_info.description.format = (VkFormat)swp->format;
		attachment_info.samples = Samples::e1;

		attachment_info.type = AttachmentRPInfo::Type::eSwapchain;
		attachment_info.swapchain = swp;
		attachment_info.should_clear = true;
		attachment_info.clear_value = c;

		Resource::Use& initial = attachment_info.initial;
		Resource::Use & final = attachment_info.final;
		// for WSI, we want to wait for colourattachmentoutput
		// we don't care about any writes, we will clear
		initial.access = vuk::AccessFlags{};
		initial.stages = vuk::PipelineStageFlagBits::eColorAttachmentOutput;
		// clear
		initial.layout = vuk::ImageLayout::ePreinitialized;
		/* Normally, we would need an external dependency at the end as well since we are changing layout in finalLayout,
   but since we are signalling a semaphore, we can rely on Vulkan's default behavior,
   which injects an external dependency here with
   dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
   dstAccessMask = 0. */
		final.access = vuk::AccessFlagBits{};
		final.layout = vuk::ImageLayout::ePresentSrcKHR;
		final.stages = vuk::PipelineStageFlagBits::eBottomOfPipe;

		bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::mark_attachment_internal(Name name, vuk::Format format, vuk::Extent2D extent, vuk::Samples samp, Clear c) {
		AttachmentRPInfo attachment_info;
		attachment_info.sizing = AttachmentRPInfo::Sizing::eAbsolute;
		attachment_info.extents = extent;
		attachment_info.iv = {};

		attachment_info.type = AttachmentRPInfo::Type::eInternal;
		attachment_info.description.format = (VkFormat)format;
		attachment_info.samples = samp;

		attachment_info.should_clear = true;
		attachment_info.clear_value = c;
		Resource::Use& initial = attachment_info.initial;
		Resource::Use & final = attachment_info.final;
		initial.access = vuk::AccessFlags{};
		initial.stages = vuk::PipelineStageFlagBits::eTopOfPipe;
		// for internal attachments we don't want to preserve previous data
		initial.layout = vuk::ImageLayout::ePreinitialized;

		// with an undefined final layout, there will be no final sync
		final.layout = vuk::ImageLayout::eUndefined;
		final.access = vuk::AccessFlagBits{};
		final.stages = vuk::PipelineStageFlagBits::eBottomOfPipe;

		bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::mark_attachment_internal(Name name, vuk::Format format, vuk::Extent2D::Framebuffer fbrel, vuk::Samples samp, Clear c) {
		AttachmentRPInfo attachment_info;
		attachment_info.sizing = AttachmentRPInfo::Sizing::eFramebufferRelative;
		attachment_info.fb_relative = fbrel;
		attachment_info.iv = {};

		attachment_info.type = AttachmentRPInfo::Type::eInternal;
		attachment_info.description.format = (VkFormat)format;
		attachment_info.samples = samp;

		attachment_info.should_clear = true;
		attachment_info.clear_value = c;
		Resource::Use& initial = attachment_info.initial;
		Resource::Use & final = attachment_info.final;
		initial.access = vuk::AccessFlags{};
		initial.stages = vuk::PipelineStageFlagBits::eTopOfPipe;
		// for internal attachments we don't want to preserve previous data
		initial.layout = vuk::ImageLayout::ePreinitialized;

		// with an undefined final layout, there will be no final sync
		final.layout = vuk::ImageLayout::eUndefined;
		final.access = vuk::AccessFlagBits{};
		final.stages = vuk::PipelineStageFlagBits::eBottomOfPipe;

		bound_attachments.emplace(name, attachment_info);
	}

	void RenderGraph::mark_attachment_resolve(Name resolved_name, Name ms_name) {
		add_pass({
			.resources = {
				vuk::Resource{ms_name, ms_name, vuk::Resource::Type::eImage, vuk::eColorResolveRead},
				vuk::Resource{resolved_name, resolved_name, vuk::Resource::Type::eImage, vuk::eColorResolveWrite}
			},
			.resolves = {{ms_name, resolved_name}}
			});
	}

	void RenderGraph::bind_buffer(Name name, vuk::Buffer buf, Access initial, Access final) {
		BufferInfo buf_info{ .name = name, .initial = to_use(initial), .final = to_use(final), .buffer = buf };
		bound_buffers.emplace(name, buf_info);
	}

	void RenderGraph::bind_attachment(Name name, Attachment att, Access initial_acc, Access final_acc) {
		AttachmentRPInfo attachment_info;
		attachment_info.sizing = AttachmentRPInfo::Sizing::eAbsolute;
		attachment_info.extents = att.extent;
		attachment_info.image = att.image;
		attachment_info.iv = att.image_view;

		attachment_info.type = AttachmentRPInfo::Type::eExternal;
		attachment_info.description.format = (VkFormat)att.format;
		attachment_info.samples = att.sample_count;

		attachment_info.should_clear = initial_acc == Access::eClear; // if initial access was clear, we will clear
		attachment_info.clear_value = att.clear_value;
		Resource::Use& initial = attachment_info.initial;
		Resource::Use & final = attachment_info.final;
		initial = to_use(initial_acc);
		final = to_use(final_acc);
		bound_attachments.emplace(name, attachment_info);
	}

	void sync_bound_attachment_to_renderpass(vuk::RenderGraph::AttachmentRPInfo& rp_att, vuk::RenderGraph::AttachmentRPInfo& attachment_info) {
		rp_att.description.format = attachment_info.description.format;
		rp_att.samples = attachment_info.samples;
		rp_att.description.samples = (VkSampleCountFlagBits)attachment_info.samples.count;
		rp_att.iv = attachment_info.iv;
		rp_att.extents = attachment_info.extents;
		rp_att.clear_value = attachment_info.clear_value;
		rp_att.should_clear = attachment_info.should_clear;
		rp_att.type = attachment_info.type;
	}

	void RenderGraph::build(vuk::PerThreadContext& ptc) {
		for (auto& [raw_name, attachment_info] : bound_attachments) {
			auto name = resolve_name(raw_name, aliases);
			auto& chain = use_chains.at(name);
			chain.insert(chain.begin(), UseRef{ std::move(attachment_info.initial), nullptr });
			chain.emplace_back(UseRef{ attachment_info.final, nullptr });

			vuk::ImageAspectFlags aspect = format_to_aspect((vuk::Format)attachment_info.description.format);

			for (size_t i = 0; i < chain.size() - 1; i++) {
				auto& left = chain[i];
				auto& right = chain[i + 1];

				bool crosses_rpass = (left.pass == nullptr || right.pass == nullptr || left.pass->render_pass_index != right.pass->render_pass_index);
				if (crosses_rpass) {
					if (left.pass) { // RenderPass ->
						auto& left_rp = rpis[left.pass->render_pass_index];
						// if this is an attachment, we specify layout
						if (is_framebuffer_attachment(left.use)) {
							assert(!left_rp.framebufferless);
							auto& rp_att = *contains_if(left_rp.attachments, [name](auto& att) {return att.name == name; });

							sync_bound_attachment_to_renderpass(rp_att, attachment_info);
							// if there is a "right" rp
							// or if this attachment has a required end layout
							// then we transition for it
							if (right.pass || right.use.layout != vuk::ImageLayout::eUndefined) {
								rp_att.description.finalLayout = (VkImageLayout)right.use.layout;
							} else {
								// we keep last use as finalLayout
								rp_att.description.finalLayout = (VkImageLayout)left.use.layout;
							}
							// compute attachment store
							if (right.use.layout == vuk::ImageLayout::eUndefined) {
								rp_att.description.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
							} else {
								rp_att.description.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
							}
						}
						// TODO: we need a dep here if there is a write or there is a layout transition
						if (/*left.use.layout != right.use.layout &&*/ right.use.layout != vuk::ImageLayout::eUndefined && !left_rp.framebufferless) { // different layouts, need to have dependency
							VkSubpassDependency sd{};
							sd.dstAccessMask = (VkAccessFlags)right.use.access;
							sd.dstStageMask = (VkPipelineStageFlags)right.use.stages;
							sd.srcSubpass = left.pass->subpass;
							sd.srcAccessMask = (VkAccessFlags)left.use.access;
							sd.srcStageMask = (VkPipelineStageFlags)left.use.stages;
							sd.dstSubpass = VK_SUBPASS_EXTERNAL;
							left_rp.rpci.subpass_dependencies.push_back(sd);
						}
						// if we are coming from an fbless pass we need to emit barriers if the right pass doesn't exist (chain end) or has framebuffer 
						if (left_rp.framebufferless && ((right.pass && !rpis[right.pass->render_pass_index].framebufferless) || right.pass == nullptr)) {
							// right layout == Undefined means the chain terminates, no transition/barrier
							if (right.use.layout == vuk::ImageLayout::eUndefined)
								continue;
							VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
							barrier.dstAccessMask = (VkAccessFlags)right.use.access;
							barrier.srcAccessMask = (VkAccessFlags)left.use.access;
							barrier.newLayout = (VkImageLayout)right.use.layout;
							barrier.oldLayout = (VkImageLayout)left.use.layout;
							barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
							barrier.subresourceRange.baseArrayLayer = 0;
							barrier.subresourceRange.baseMipLevel = 0;
							barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
							barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
							ImageBarrier ib{ .image = name, .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
							left_rp.subpasses[left.pass->subpass].post_barriers.push_back(ib);
						}
					}

					if (right.pass) { // -> RenderPass
						auto& right_rp = rpis[right.pass->render_pass_index];
						// if this is an attachment, we specify layout
						if (is_framebuffer_attachment(right.use)) {
							assert(!right_rp.framebufferless);
							auto& rp_att = *contains_if(right_rp.attachments, [name](auto& att) {return att.name == name; });

							sync_bound_attachment_to_renderpass(rp_att, attachment_info);
							// we will have "left" transition for us
							if (left.pass) {
								rp_att.description.initialLayout = (VkImageLayout)right.use.layout;
							} else {
								// if there is no "left" renderpass, then we take the initial layout
								rp_att.description.initialLayout = (VkImageLayout)left.use.layout;
							}
							// compute attachment load
							if (left.use.layout == vuk::ImageLayout::eUndefined) {
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
							} else if (left.use.layout == vuk::ImageLayout::ePreinitialized) {
								// preinit means clear
								rp_att.description.initialLayout = (VkImageLayout)vuk::ImageLayout::eUndefined;
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
							} else {
								rp_att.description.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
							}
						}
						// TODO: we need a dep here if there is a write or there is a layout transition
						if (/*right.use.layout != left.use.layout &&*/ left.use.layout != vuk::ImageLayout::eUndefined) { // different layouts, need to have dependency
							VkSubpassDependency sd{};
							sd.dstAccessMask = (VkAccessFlags)right.use.access;
							sd.dstStageMask = (VkPipelineStageFlags)right.use.stages;
							sd.dstSubpass = right.pass->subpass;
							sd.srcAccessMask = (VkAccessFlags)left.use.access;
							sd.srcStageMask = (VkPipelineStageFlags)left.use.stages;
							sd.srcSubpass = VK_SUBPASS_EXTERNAL;
							right_rp.rpci.subpass_dependencies.push_back(sd);
						}
						if (right_rp.framebufferless) {
							if (left.pass) {
								auto& left_rp = rpis[left.pass->render_pass_index];
								// if we are coming from a renderpass and this was a framebuffer attachment there
								// then the renderpass transitioned this resource for us, and we don't need a barrier
								if (!left_rp.framebufferless && is_framebuffer_attachment(left.use))
									continue;
							}
							VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
							barrier.dstAccessMask = (VkAccessFlags)right.use.access;
							barrier.srcAccessMask = (VkAccessFlags)left.use.access;
							barrier.newLayout = (VkImageLayout)right.use.layout;
							barrier.oldLayout = left.use.layout == vuk::ImageLayout::ePreinitialized ? (VkImageLayout)vuk::ImageLayout::eUndefined : (VkImageLayout)left.use.layout;
							barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
							barrier.subresourceRange.baseArrayLayer = 0;
							barrier.subresourceRange.baseMipLevel = 0;
							barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
							barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
							ImageBarrier ib{ .image = name, .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
							right_rp.subpasses[right.pass->subpass].pre_barriers.push_back(ib);
						}

					}
				} else { // subpass-subpass link -> subpass - subpass dependency
					// WAW, WAR, RAW accesses need sync

					// if we merged the passes into a subpass, no sync is needed
					if (left.pass->subpass == right.pass->subpass)
						continue;
					if (is_framebuffer_attachment(left.use) && (is_write_access(left.use) || (is_read_access(left.use) && is_write_access(right.use)))) {
						assert(left.pass->render_pass_index == right.pass->render_pass_index);
						auto& rp = rpis[right.pass->render_pass_index];
						VkSubpassDependency sd{};
						sd.dstAccessMask = (VkAccessFlags)right.use.access;
						sd.dstStageMask = (VkPipelineStageFlags)right.use.stages;
						sd.dstSubpass = right.pass->subpass;
						sd.srcAccessMask = (VkAccessFlags)left.use.access;
						sd.srcStageMask = (VkPipelineStageFlags)left.use.stages;
						sd.srcSubpass = left.pass->subpass;
						rp.rpci.subpass_dependencies.push_back(sd);
					}
					auto& left_rp = rpis[left.pass->render_pass_index];
					if (left_rp.framebufferless) {
						// right layout == Undefined means the chain terminates, no transition/barrier
						if (right.use.layout == vuk::ImageLayout::eUndefined)
							continue;
						VkImageMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
						barrier.srcAccessMask = (VkAccessFlags)left.use.access;
						barrier.newLayout = (VkImageLayout)right.use.layout;
						barrier.oldLayout = (VkImageLayout)left.use.layout;
						barrier.subresourceRange.aspectMask = (VkImageAspectFlags)aspect;
						barrier.subresourceRange.baseArrayLayer = 0;
						barrier.subresourceRange.baseMipLevel = 0;
						barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;
						barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
						ImageBarrier ib{ .image = name, .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
						left_rp.subpasses[left.pass->subpass].post_barriers.push_back(ib);
					}
				}
			}
		}

		for (auto& [raw_name, buffer_info] : bound_buffers) {
			auto name = resolve_name(raw_name, aliases);
			auto& chain = use_chains.at(name);
			chain.insert(chain.begin(), UseRef{ std::move(buffer_info.initial), nullptr });
			chain.emplace_back(UseRef{ buffer_info.final, nullptr });

			for (size_t i = 0; i < chain.size() - 1; i++) {
				auto& left = chain[i];
				auto& right = chain[i + 1];

				bool crosses_rpass = (left.pass == nullptr || right.pass == nullptr || left.pass->render_pass_index != right.pass->render_pass_index);
				if (crosses_rpass) {
					if (left.pass && right.use.layout != vuk::ImageLayout::eUndefined) { // RenderPass ->
						auto& left_rp = rpis[left.pass->render_pass_index];

						VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
						barrier.srcAccessMask = (VkAccessFlags)left.use.access;
						MemoryBarrier mb{ .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
						left_rp.subpasses[left.pass->subpass].post_mem_barriers.push_back(mb);
					}

					if (right.pass && left.use.layout != vuk::ImageLayout::eUndefined) { // -> RenderPass
						auto& right_rp = rpis[right.pass->render_pass_index];

						VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
						barrier.srcAccessMask = (VkAccessFlags)left.use.access;
						MemoryBarrier mb{ .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
						if (mb.src == vuk::PipelineStageFlags{}) {
							mb.src = vuk::PipelineStageFlagBits::eTopOfPipe;
							mb.barrier.srcAccessMask = {};
						}
						right_rp.subpasses[right.pass->subpass].pre_mem_barriers.push_back(mb);
					}
				} else { // subpass-subpass link -> subpass - subpass dependency
					if (left.pass->subpass == right.pass->subpass)
						continue;
					auto& left_rp = rpis[left.pass->render_pass_index];
					if (left_rp.framebufferless) {
						VkMemoryBarrier barrier{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER };
						barrier.dstAccessMask = (VkAccessFlags)right.use.access;
						barrier.srcAccessMask = (VkAccessFlags)left.use.access;
						MemoryBarrier mb{ .barrier = barrier, .src = left.use.stages, .dst = right.use.stages };
						left_rp.subpasses[left.pass->subpass].post_mem_barriers.push_back(mb);
					}
				}
			}
		}


		for (auto& rp : rpis) {
			rp.rpci.color_ref_offsets.resize(rp.subpasses.size());
			rp.rpci.ds_refs.resize(rp.subpasses.size());
		}

		// we now have enough data to build vk::RenderPasses and vk::Framebuffers
		// we have to assign the proper attachments to proper slots
		// the order is given by the resource binding order
		size_t previous_rp = -1;
		uint32_t previous_sp = -1;
		for (auto& pass : passes) {
			auto& rp = rpis[pass.render_pass_index];
			auto subpass_index = pass.subpass;
			auto& color_attrefs = rp.rpci.color_refs;
			auto& resolve_attrefs = rp.rpci.resolve_refs;
			auto& color_ref_offsets = rp.rpci.color_ref_offsets;
			auto& ds_attrefs = rp.rpci.ds_refs;

			// do not process merged passes
			if (previous_rp != -1 && previous_rp == pass.render_pass_index && previous_sp == pass.subpass) {
				continue;
			} else {
				previous_rp = pass.render_pass_index;
				previous_sp = pass.subpass;
			}

			for (auto& res : pass.pass.resources) {
				if (!is_framebuffer_attachment(res))
					continue;
				if (res.ia == vuk::Access::eColorResolveWrite) // resolve attachment are added when processing the color attachment
					continue;
				VkAttachmentReference attref{};

				auto name = resolve_name(res.use_name, aliases);
				auto& chain = use_chains.find(name)->second;
				auto cit = std::find_if(chain.begin(), chain.end(), [&](auto& useref) { return useref.pass == &pass; });
				assert(cit != chain.end());
				attref.layout = (VkImageLayout)cit->use.layout;
				attref.attachment = (uint32_t)std::distance(rp.attachments.begin(), std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) { return name == att.name; }));

				if (attref.layout != (VkImageLayout)vuk::ImageLayout::eColorAttachmentOptimal) {
					if (attref.layout == (VkImageLayout)vuk::ImageLayout::eDepthStencilAttachmentOptimal) {
						ds_attrefs[subpass_index] = attref;
					}
				} else {

					VkAttachmentReference rref{};
					rref.attachment = VK_ATTACHMENT_UNUSED;
					if (auto it = pass.pass.resolves.find(res.use_name); it != pass.pass.resolves.end()) {
						// this a resolve src attachment
						// get the dst attachment
						auto& dst_name = it->second;
						rref.layout = (VkImageLayout)vuk::ImageLayout::eColorAttachmentOptimal; // the only possible layout for resolves
						rref.attachment = (uint32_t)std::distance(rp.attachments.begin(), std::find_if(rp.attachments.begin(), rp.attachments.end(), [&](auto& att) { return dst_name == att.name; }));
					}

					// we insert the new attachment at the end of the list for current subpass index
					if (subpass_index < rp.subpasses.size() - 1) {
						auto next_start = color_ref_offsets[subpass_index + 1];
						color_attrefs.insert(color_attrefs.begin() + next_start, attref);
						resolve_attrefs.insert(resolve_attrefs.begin() + next_start, rref);
					} else {
						color_attrefs.push_back(attref);
						resolve_attrefs.push_back(rref);
					}
					for (size_t i = subpass_index + 1; i < rp.subpasses.size(); i++) {
						color_ref_offsets[i]++;
					}
				}
			}
		}

		for (auto& rp : rpis) {
			if (rp.attachments.size() == 0) {
				continue;
			}

			auto& subp = rp.rpci.subpass_descriptions;
			auto& color_attrefs = rp.rpci.color_refs;
			auto& color_ref_offsets = rp.rpci.color_ref_offsets;
			auto& resolve_attrefs = rp.rpci.resolve_refs;
			auto& ds_attrefs = rp.rpci.ds_refs;

			// subpasses
			for (size_t i = 0; i < rp.subpasses.size(); i++) {
				vuk::SubpassDescription sd;
				size_t color_count = 0;
				if (i < rp.subpasses.size() - 1) {
					color_count = color_ref_offsets[i + 1] - color_ref_offsets[i];
				} else {
					color_count = color_attrefs.size() - color_ref_offsets[i];
				}
				{
					auto first = color_attrefs.data() + color_ref_offsets[i];
					sd.colorAttachmentCount = (uint32_t)color_count;
					sd.pColorAttachments = first;
				}

				sd.pDepthStencilAttachment = ds_attrefs[i] ? &*ds_attrefs[i] : nullptr;
				sd.flags = {};
				sd.inputAttachmentCount = 0;
				sd.pInputAttachments = nullptr;
				sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
				sd.preserveAttachmentCount = 0;
				sd.pPreserveAttachments = nullptr;
				{
					auto first = resolve_attrefs.data() + color_ref_offsets[i];
					sd.pResolveAttachments = first;
				}

				subp.push_back(sd);
			}

			rp.rpci.subpassCount = (uint32_t)rp.rpci.subpass_descriptions.size();
			rp.rpci.pSubpasses = rp.rpci.subpass_descriptions.data();

			rp.rpci.dependencyCount = (uint32_t)rp.rpci.subpass_dependencies.size();
			rp.rpci.pDependencies = rp.rpci.subpass_dependencies.data();

			// attachments
			vuk::Samples samples(vuk::SampleCountFlagBits::e1);
			for (auto& attrpinfo : rp.attachments) {
				auto& bound = bound_attachments.at(attrpinfo.name);
				if (!bound.samples.infer) {
					samples = bound.samples;
				}
			}
			for (auto& attrpinfo : rp.attachments) {
				if (attrpinfo.samples.infer) {
					attrpinfo.description.samples = (VkSampleCountFlagBits)samples.count;
				} else {
					attrpinfo.description.samples = (VkSampleCountFlagBits)attrpinfo.samples.count;
				}
				rp.rpci.attachments.push_back(attrpinfo.description);
			}

			rp.rpci.attachmentCount = (uint32_t)rp.rpci.attachments.size();
			rp.rpci.pAttachments = rp.rpci.attachments.data();

			rp.handle = ptc.acquire_renderpass(rp.rpci);
		}
	}

	void RenderGraph::create_attachment(PerThreadContext& ptc, Name name, RenderGraph::AttachmentRPInfo& attachment_info, vuk::Extent2D fb_extent, vuk::SampleCountFlagBits samples) {
		auto& chain = use_chains.at(name);
		if (attachment_info.type == AttachmentRPInfo::Type::eInternal) {
			vuk::ImageUsageFlags usage = compute_usage(chain);

			vuk::ImageCreateInfo ici;
			ici.usage = usage;
			ici.arrayLayers = 1;
			// compute extent
			if (attachment_info.sizing == AttachmentRPInfo::Sizing::eFramebufferRelative) {
				assert(fb_extent.width > 0 && fb_extent.height > 0);
				ici.extent = vuk::Extent3D{ static_cast<uint32_t>(attachment_info.fb_relative.width * fb_extent.width), static_cast<uint32_t>(attachment_info.fb_relative.height * fb_extent.height), 1u };
			} else {
				ici.extent = static_cast<vuk::Extent3D>(attachment_info.extents);
			}
			ici.imageType = vuk::ImageType::e2D;
			ici.format = vuk::Format(attachment_info.description.format);
			ici.mipLevels = 1;
			ici.initialLayout = vuk::ImageLayout::eUndefined;
			ici.samples = samples;
			ici.sharingMode = vuk::SharingMode::eExclusive;
			ici.tiling = vuk::ImageTiling::eOptimal;

			vuk::ImageViewCreateInfo ivci;
			ivci.image = vuk::Image{};
			ivci.format = vuk::Format(attachment_info.description.format);
			ivci.viewType = vuk::ImageViewType::e2D;
			vuk::ImageSubresourceRange isr;

			isr.aspectMask = format_to_aspect(ici.format);
			isr.baseArrayLayer = 0;
			isr.layerCount = 1;
			isr.baseMipLevel = 0;
			isr.levelCount = 1;
			ivci.subresourceRange = isr;

			RGCI rgci;
			rgci.name = name;
			rgci.ici = ici;
			rgci.ivci = ivci;

			auto rg = ptc.acquire_rendertarget(rgci);
			attachment_info.iv = rg.image_view;
			attachment_info.image = rg.image;
		}
	}

	void begin_renderpass(vuk::RenderGraph::RenderPassInfo& rpass, VkCommandBuffer& cbuf, bool use_secondary_command_buffers) {
		if (rpass.handle == VK_NULL_HANDLE) {
			return;
		}

		VkRenderPassBeginInfo rbi{ .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
		rbi.renderPass = rpass.handle;
		rbi.framebuffer = rpass.framebuffer;
		rbi.renderArea = vuk::Rect2D{ vuk::Offset2D{}, vuk::Extent2D{ rpass.fbci.width, rpass.fbci.height } };
		std::vector<VkClearValue> clears;
		for (size_t i = 0; i < rpass.attachments.size(); i++) {
			auto& att = rpass.attachments[i];
			if (att.should_clear)
				clears.push_back(att.clear_value.c);
		}
		rbi.pClearValues = clears.data();
		rbi.clearValueCount = (uint32_t)clears.size();

		vkCmdBeginRenderPass(cbuf, &rbi, use_secondary_command_buffers ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
	}

	void RenderGraph::fill_renderpass_info(vuk::RenderGraph::RenderPassInfo& rpass, const size_t& i, vuk::CommandBuffer& cobuf) {
		if (rpass.handle == VK_NULL_HANDLE) {
			cobuf.ongoing_renderpass = {};
			return;
		}
		vuk::CommandBuffer::RenderPassInfo rpi;
		rpi.renderpass = rpass.handle;
		rpi.subpass = (uint32_t)i;
		rpi.extent = vuk::Extent2D{ rpass.fbci.width, rpass.fbci.height };
		auto& spdesc = rpass.rpci.subpass_descriptions[i];
		rpi.color_attachments = std::span<const VkAttachmentReference>(spdesc.pColorAttachments, spdesc.colorAttachmentCount);
		for (auto& ca : rpi.color_attachments) {
			auto& att = rpass.attachments[ca.attachment];
			if (!att.samples.infer)
				rpi.samples = att.samples.count;
		}
		// TODO: depth could be msaa too
		if (rpi.color_attachments.size() == 0) { // depth only pass, samples == 1
			rpi.samples = vuk::SampleCountFlagBits::e1;
		}
		cobuf.ongoing_renderpass = rpi;
	}

	VkCommandBuffer RenderGraph::execute(vuk::PerThreadContext& ptc, std::vector<std::pair<SwapChainRef, size_t>> swp_with_index) {
		// create framebuffers, create & bind attachments
		for (auto& rp : rpis) {
			if (rp.attachments.size() == 0)
				continue;

			vuk::Extent2D fb_extent;
			bool extent_known = false;

			// bind swapchain attachments, deduce framebuffer size & sample count
			for (auto& attrpinfo : rp.attachments) {
				auto& bound = bound_attachments[attrpinfo.name];

				if (bound.type == AttachmentRPInfo::Type::eSwapchain) {
					auto it = std::find_if(swp_with_index.begin(), swp_with_index.end(), [&](auto& t) { return t.first == bound.swapchain; });
					bound.iv = it->first->image_views[it->second];
					bound.image = it->first->images[it->second];
					fb_extent = it->first->extent;
					extent_known = true;
				} else {
					if (bound.sizing == AttachmentRPInfo::Sizing::eAbsolute) {
						fb_extent = bound.extents;
						extent_known = true;
					}
				}
			}

			for (auto& attrpinfo : rp.attachments) {
				auto& bound = bound_attachments[attrpinfo.name];
				if (extent_known) {
					bound.extents = fb_extent;
				}
			}
		}

		for (auto& [name, bound] : bound_attachments) {
			if (bound.type == AttachmentRPInfo::Type::eSwapchain) {
				auto it = std::find_if(swp_with_index.begin(), swp_with_index.end(), [boundb = &bound](auto& t) { return t.first == boundb->swapchain; });
				bound.iv = it->first->image_views[it->second];
				bound.image = it->first->images[it->second];
			}
		}

		for (auto& rp : rpis) {
			if (rp.attachments.size() == 0)
				continue;

			auto& ivs = rp.fbci.attachments;
			std::vector<VkImageView> vkivs;
			vuk::Extent2D fb_extent;

			for (auto& attrpinfo : rp.attachments) {
				auto& bound = bound_attachments[attrpinfo.name];
				if (bound.extents.width > 0 && bound.extents.height > 0)
					fb_extent = bound.extents;
			}

			// create internal attachments; bind attachments to fb
			for (auto& attrpinfo : rp.attachments) {
				auto& bound = bound_attachments[attrpinfo.name];
				if (bound.type == AttachmentRPInfo::Type::eInternal) {
					create_attachment(ptc, attrpinfo.name, bound, fb_extent, (vuk::SampleCountFlagBits)attrpinfo.description.samples);
				}

				ivs.push_back(bound.iv);
				vkivs.push_back(bound.iv.payload);
			}
			rp.fbci.renderPass = rp.handle;
			rp.fbci.width = fb_extent.width;
			rp.fbci.height = fb_extent.height;
			rp.fbci.pAttachments = &vkivs[0];
			rp.fbci.attachmentCount = (uint32_t)vkivs.size();
			rp.fbci.layers = 1;
			rp.framebuffer = ptc.acquire_framebuffer(rp.fbci);
		}
		// create non-attachment images
		for (auto& [name, bound] : bound_attachments) {
			if (bound.type == AttachmentRPInfo::Type::eInternal && bound.image == VK_NULL_HANDLE) {
				create_attachment(ptc, name, bound, vuk::Extent2D{ 0,0 }, bound.samples.count);
			}
		}

		// actual execution
		auto cbuf = ptc.acquire_command_buffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY);

		VkCommandBufferBeginInfo cbi{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
		vkBeginCommandBuffer(cbuf, &cbi);

		CommandBuffer cobuf(*this, ptc, cbuf);
		for (auto& rpass : rpis) {
			bool use_secondary_command_buffers = rpass.subpasses[0].use_secondary_command_buffers;
			begin_renderpass(rpass, cbuf, use_secondary_command_buffers);
			for (size_t i = 0; i < rpass.subpasses.size(); i++) {
				auto& sp = rpass.subpasses[i];
				fill_renderpass_info(rpass, i, cobuf);
				// insert image pre-barriers
				if (rpass.handle == VK_NULL_HANDLE) {
					for (auto dep : sp.pre_barriers) {
						dep.barrier.image = bound_attachments[dep.image].image;
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
					}
					for (auto dep : sp.pre_mem_barriers) {
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
					}
				}
				for (auto& p : sp.passes) {
					// if pass requested no secondary cbufs, but due to subpass merging that is what we got
					if (p->pass.use_secondary_command_buffers == false && use_secondary_command_buffers == true) {
						auto secondary = cobuf.begin_secondary();
						if (p->pass.execute) {
							secondary.current_pass = p;
							if (!p->pass.name.empty()) {
								//ptc.ctx.debug.begin_region(cobuf.command_buffer, sp.pass->pass.name);
								p->pass.execute(secondary);
								//ptc.ctx.debug.end_region(cobuf.command_buffer);
							} else {
								p->pass.execute(secondary);
							}
						}
						auto result = secondary.get_buffer();
						cobuf.execute({ &result, 1 });
					} else {
						if (p->pass.execute) {
							cobuf.current_pass = p;
							if (!p->pass.name.empty()) {
								//ptc.ctx.debug.begin_region(cobuf.command_buffer, sp.pass->pass.name);
								p->pass.execute(cobuf);
								//ptc.ctx.debug.end_region(cobuf.command_buffer);
							} else {
								p->pass.execute(cobuf);
							}
						}

						cobuf.attribute_descriptions.clear();
						cobuf.binding_descriptions.clear();
						cobuf.set_bindings = {};
						cobuf.sets_used = {};
					}
				}
				if (i < rpass.subpasses.size() - 1 && rpass.handle != VK_NULL_HANDLE) {
					use_secondary_command_buffers = rpass.subpasses[i + 1].use_secondary_command_buffers;
					vkCmdNextSubpass(cbuf, use_secondary_command_buffers ? VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS : VK_SUBPASS_CONTENTS_INLINE);
				}

				// insert image post-barriers
				if (rpass.handle == VK_NULL_HANDLE) {
					for (auto dep : sp.post_barriers) {
						dep.barrier.image = bound_attachments[dep.image].image;
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 0, nullptr, 0, nullptr, 1, &dep.barrier);
					}
					for (auto dep : sp.post_mem_barriers) {
						vkCmdPipelineBarrier(cbuf, (VkPipelineStageFlags)dep.src, (VkPipelineStageFlags)dep.dst, 0, 1, &dep.barrier, 0, nullptr, 0, nullptr);
					}
				}
			}
			if (rpass.handle != VK_NULL_HANDLE) {
				vkCmdEndRenderPass(cbuf);
			}
		}
		vkEndCommandBuffer(cbuf);
		return cbuf;
	}

	RenderGraph::BufferInfo RenderGraph::get_resource_buffer(Name n) {
		return bound_buffers.at(n);
	}

	bool RenderGraph::is_resource_image_in_general_layout(Name n, PassInfo* pass_info) {
		auto& chain = use_chains.at(n);
		for (auto& elem : chain) {
			if (elem.pass == pass_info) {
				return elem.use.layout == vuk::ImageLayout::eGeneral;
			}
		}
		assert(0);
		return false;
	}

	RenderGraph::RenderPassInfo::RenderPassInfo(arena& arena_) : INIT2(subpasses), INIT2(attachments) {
	}

	PassInfo::PassInfo(arena& arena_, Pass&& p) : pass(std::move(p)) {}

	RenderGraph::SubpassInfo::SubpassInfo(arena& arena_) : INIT2(passes) {}
#undef INIT
} // namespace vuk
