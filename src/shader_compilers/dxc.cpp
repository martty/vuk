#ifdef _WIN32
// dxcapi.h expects the COM API to be present on Windows.
// On other platforms, the Vulkan SDK will have WinAdapter.h alongside dxcapi.h that is automatically included to stand in for the COM API.
#include <atlbase.h>
#endif

#include "vuk/Exception.hpp"
#include "vuk/Result.hpp"
#include "vuk/ShaderSource.hpp"

#include <dxc/dxcapi.h>
#define DXC_HR(hr, msg)                                                                                                                                        \
	if (FAILED(hr)) {                                                                                                                                            \
		throw ShaderCompilationException{ msg };                                                                                                                   \
	}

namespace {
	std::wstring convert_to_wstring(const std::string& string) {
		std::vector<wchar_t> buffer(string.size());
		std::use_facet<std::ctype<wchar_t>>(std::locale()).widen(string.data(), string.data() + string.size(), buffer.data());
		return { buffer.data(), buffer.size() };
	}
}

namespace vuk {
	Result<std::vector<uint32_t>> compile_hlsl(const ShaderModuleCreatInfo& cinfo, uint32_t shader_compiler_target_version) {
		std::vector<LPCWSTR> arguments;
		arguments.push_back(L"-E");

		auto entry_point = convert_to_wstring(cinfo.source.entry_point);
		arguments.push_back(entry_point.c_str());

		auto dir = std::filesystem::path(cinfo.filename).parent_path();
		auto include_path = fmt::format("-I {0}", dir.string());
		auto include_path_w = convert_to_wstring(include_path);
		arguments.push_back(include_path_w.c_str());

		std::vector<std::wstring> def_ws;
		for (auto [k, v] : cinfo.defines) {
			auto def = v.empty() ? fmt::format("-D{0}", k) : fmt::format("-D{0}={1}", k, v);
			arguments.push_back(def_ws.emplace_back(convert_to_wstring(def)).c_str());
		}

		// current valid options in dxc are 1.0 and 1.1
		static const std::unordered_map<uint32_t, const wchar_t*> target_version = {
			{ VK_API_VERSION_1_0, L"-fspv-target-env=vulkan1.0" },
			{ VK_API_VERSION_1_1, L"-fspv-target-env=vulkan1.1" },
			{ VK_API_VERSION_1_2, L"-fspv-target-env=vulkan1.1" },
			{ VK_API_VERSION_1_3, L"-fspv-target-env=vulkan1.1" },
		};

		arguments.push_back(target_version.at(shader_compiler_target_version));

		static const std::unordered_map<ShaderCompileOptions::OptimizationLevel, const wchar_t*> optimization_level = {
			{ ShaderCompileOptions::OptimizationLevel::O0, L"-O0" },
			{ ShaderCompileOptions::OptimizationLevel::O1, L"-O1" },
			{ ShaderCompileOptions::OptimizationLevel::O2, L"-O2" },
			{ ShaderCompileOptions::OptimizationLevel::O3, L"-O3" },
		};

		arguments.push_back(optimization_level.at(cinfo.compile_options.optimization_level));

		for (auto& arg : cinfo.compile_options.dxc_extra_arguments) {
			arguments.push_back(arg.data());
		}

		static const std::pair<const char*, HlslShaderStage> inferred[] = { { ".vert.", HlslShaderStage::eVertex },  { ".frag.", HlslShaderStage::ePixel },
			                                                                  { ".comp.", HlslShaderStage::eCompute }, { ".geom.", HlslShaderStage::eGeometry },
			                                                                  { ".mesh.", HlslShaderStage::eMesh },    { ".hull.", HlslShaderStage::eHull },
			                                                                  { ".dom.", HlslShaderStage::eDomain },   { ".amp.", HlslShaderStage::eAmplification } };

		static const std::unordered_map<HlslShaderStage, LPCWSTR> stage_mappings{
			{ HlslShaderStage::eVertex, L"vs_6_7" },   { HlslShaderStage::ePixel, L"ps_6_7" },        { HlslShaderStage::eCompute, L"cs_6_7" },
			{ HlslShaderStage::eGeometry, L"gs_6_7" }, { HlslShaderStage::eMesh, L"ms_6_7" },         { HlslShaderStage::eHull, L"hs_6_7" },
			{ HlslShaderStage::eDomain, L"ds_6_7" },   { HlslShaderStage::eAmplification, L"as_6_7" }
		};

		HlslShaderStage shader_stage = cinfo.source.hlsl_stage;
		if (shader_stage == HlslShaderStage::eInferred) {
			for (const auto& [ext, stage] : inferred) {
				if (cinfo.filename.find(ext) != std::string::npos) {
					shader_stage = stage;
					break;
				}
			}
		}

		assert((shader_stage != HlslShaderStage::eInferred) && "Failed to infer HLSL shader stage");

		arguments.push_back(L"-T");
		arguments.push_back(stage_mappings.at(shader_stage));

		DxcBuffer source_buf;
		source_buf.Ptr = cinfo.source.as_c_str();
		source_buf.Size = cinfo.source.data.size() * 4;
		source_buf.Encoding = 0;

		CComPtr<IDxcCompiler3> compiler = nullptr;
		DXC_HR(DxcCreateInstance(CLSID_DxcCompiler, __uuidof(IDxcCompiler3), (void**)&compiler), "Failed to create DXC compiler");

		CComPtr<IDxcUtils> utils = nullptr;
		DXC_HR(DxcCreateInstance(CLSID_DxcUtils, __uuidof(IDxcUtils), (void**)&utils), "Failed to create DXC utils");

		CComPtr<IDxcIncludeHandler> include_handler = nullptr;
		DXC_HR(utils->CreateDefaultIncludeHandler(&include_handler), "Failed to create include handler");

		CComPtr<IDxcResult> result = nullptr;
		DXC_HR(compiler->Compile(&source_buf, arguments.data(), (UINT32)arguments.size(), &*include_handler, __uuidof(IDxcResult), (void**)&result),
		       "Failed to compile with DXC");

		CComPtr<IDxcBlobUtf8> errors = nullptr;
		DXC_HR(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr), "Failed to get DXC compile errors");
		if (errors && errors->GetStringLength() > 0) {
			std::string message = errors->GetStringPointer();
			throw ShaderCompilationException{ message };
		}

		CComPtr<IDxcBlob> output = nullptr;
		DXC_HR(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&output), nullptr), "Failed to get DXC output");
		assert(output != nullptr);

		const uint32_t* begin = (const uint32_t*)output->GetBufferPointer();
		const uint32_t* end = begin + (output->GetBufferSize() / 4);

		return std::vector<uint32_t>{ begin, end };
	}
} // namespace vuk