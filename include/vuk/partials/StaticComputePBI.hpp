#pragma once

#include "vuk/Pipeline.hpp"

#include <string>
#include <cstdlib>

namespace vuk {
	namespace detail {
		// helper for SPIRVTemplates
		inline PipelineBaseInfo* static_compute_pbi(Context& ctx, const uint32_t* ptr, size_t size, std::string ident) {
			vuk::PipelineBaseCreateInfo pci;
			std::vector<uint32_t> spirv_bytes(size);
			std::copy(ptr, ptr + size, spirv_bytes.begin());
			pci.add_spirv(std::move(spirv_bytes), std::move(ident));
			/*FILE* fo = fopen("dumb.spv", "wb");
			fwrite(ptr, sizeof(uint32_t), size, fo);
			fclose(fo);*/
			return ctx.get_pipeline(pci);
		}
	} // namespace detail
} // namespace vuk