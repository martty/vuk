#include "fmt/chrono.h"
#include "source/core/slang-list.h"
#include "vuk/Exception.hpp"
#include "vuk/Result.hpp"
#include "vuk/ShaderSource.hpp"

#include <filesystem>
#include <slang-com-ptr.h>

#define CHECK_RESULT(x)                                                                                                                                        \
	if (SLANG_FAILED(SlangResult(x))) {                                                                                                                          \
		auto err = fmt::format("Slang error: {}", x);                                                                                                              \
		return { expected_error, ShaderCompilationException(err) };                                                                                                \
	}

#define CHECK_DIAGNOSTIC(diagnosticsBlob)                                                                                                                      \
	if ((diagnosticsBlob) != nullptr) {                                                                                                                          \
		auto err = fmt::format("{}", (const char*)(diagnosticsBlob)->getBufferPointer());                                                                          \
		return { expected_error, ShaderCompilationException(err) };                                                                                                \
	}

namespace vuk {
	Result<std::vector<uint32_t>> compile_slang(const ShaderModuleCreateInfo& cinfo, uint32_t shader_compiler_target_version) {
		Slang::ComPtr<slang::IGlobalSession> slangGlobalSession;
		CHECK_RESULT(slang::createGlobalSession(slangGlobalSession.writeRef()))

		slang::SessionDesc sessionDesc = {};
		slang::TargetDesc targetDesc = {};
		targetDesc.format = SLANG_SPIRV;
		targetDesc.profile = slangGlobalSession->findProfile("glsl440");
		targetDesc.flags = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY;

		std::vector<slang::CompilerOptionEntry> entries = {};
		entries.emplace_back(slang::CompilerOptionName::VulkanUseEntryPointName, slang::CompilerOptionValue{ .intValue0 = true });

		slang::CompilerOptionEntry opt_level_entry;
		opt_level_entry.name = slang::CompilerOptionName::Optimization;
		switch (cinfo.compile_options.optimization_level) {
		case ShaderCompileOptions::OptimizationLevel::O0:
			opt_level_entry.value.intValue0 = SLANG_OPTIMIZATION_LEVEL_NONE;
			break;
		case ShaderCompileOptions::OptimizationLevel::O1:
			opt_level_entry.value.intValue0 = SLANG_OPTIMIZATION_LEVEL_DEFAULT;
			break;
		case ShaderCompileOptions::OptimizationLevel::O2:
			opt_level_entry.value.intValue0 = SLANG_OPTIMIZATION_LEVEL_HIGH;
			break;
		case ShaderCompileOptions::OptimizationLevel::O3:
			opt_level_entry.value.intValue0 = SLANG_OPTIMIZATION_LEVEL_MAXIMAL;
			break;
		}
		entries.emplace_back(opt_level_entry);

		if (cinfo.compile_options.compiler_flags & ShaderCompilerFlagBits::eGlLayout)
			entries.emplace_back(slang::CompilerOptionName::VulkanUseGLLayout, slang::CompilerOptionValue{ .intValue0 = true });

		if (cinfo.compile_options.compiler_flags & ShaderCompilerFlagBits::eNoWarnings)
			entries.emplace_back(slang::CompilerOptionName::DisableWarnings, slang::CompilerOptionValue{ .intValue0 = true });
		else if (cinfo.compile_options.compiler_flags & ShaderCompilerFlagBits::eWarningsAsErrors)
			entries.emplace_back(slang::CompilerOptionName::WarningsAsErrors, slang::CompilerOptionValue{ .intValue0 = true });

		if (cinfo.compile_options.compiler_flags & ShaderCompilerFlagBits::eMatrixColumnMajor)
			entries.emplace_back(slang::CompilerOptionName::MatrixLayoutColumn, slang::CompilerOptionValue{ .intValue0 = true });
		else if (cinfo.compile_options.compiler_flags & ShaderCompilerFlagBits::eMatrixRowMajor)
			entries.emplace_back(slang::CompilerOptionName::MatrixLayoutRow, slang::CompilerOptionValue{ .intValue0 = true });

		if (cinfo.compile_options.compiler_flags & ShaderCompilerFlagBits::eInvertY)
			entries.emplace_back(slang::CompilerOptionName::VulkanInvertY, slang::CompilerOptionValue{ .intValue0 = true });

		if (cinfo.compile_options.compiler_flags & ShaderCompilerFlagBits::eDxPositionW)
			entries.emplace_back(slang::CompilerOptionName::VulkanUseDxPositionW, slang::CompilerOptionValue{ .intValue0 = true });

		targetDesc.compilerOptionEntries = entries.data();
		targetDesc.compilerOptionEntryCount = (uint32_t)entries.size();

		sessionDesc.targets = &targetDesc;
		sessionDesc.targetCount = 1;

		Slang::ComPtr<slang::ISession> session;
		CHECK_RESULT(slangGlobalSession->createSession(sessionDesc, session.writeRef()))

		slang::IModule* slangModule;
		{
			Slang::ComPtr<slang::IBlob> diagnosticBlob;
			slangModule = session->loadModule(cinfo.filename.c_str(), diagnosticBlob.writeRef());
			CHECK_DIAGNOSTIC(diagnosticBlob)
			if (!slangModule)
				return { expected_error, ShaderCompilationException("Couldn't load the module!") };
		}

		Slang::ComPtr<slang::IEntryPoint> entryPoint;
		slangModule->findEntryPointByName(cinfo.source.entry_point.c_str(), entryPoint.writeRef());

		Slang::List<slang::IComponentType*> componentTypes;
		componentTypes.add(slangModule);
		componentTypes.add(entryPoint);

		Slang::ComPtr<slang::IComponentType> composedProgram;
		{
			Slang::ComPtr<slang::IBlob> diagnosticsBlob;
			const SlangResult result =
			    session->createCompositeComponentType(componentTypes.getBuffer(), componentTypes.getCount(), composedProgram.writeRef(), diagnosticsBlob.writeRef());
			CHECK_DIAGNOSTIC(diagnosticsBlob)
			CHECK_RESULT(result)
		}

		Slang::ComPtr<slang::IBlob> spirvCode;
		{
			Slang::ComPtr<slang::IBlob> diagnosticsBlob;
			const SlangResult result = composedProgram->getEntryPointCode(0, 0, spirvCode.writeRef(), diagnosticsBlob.writeRef());
			CHECK_DIAGNOSTIC(diagnosticsBlob)
			CHECK_RESULT(result)
		}

		const uint32_t* begin = (const uint32_t*)spirvCode->getBufferPointer();
		const uint32_t* end = begin + spirvCode->getBufferSize() / 4;

		return { expected_value, std::vector<uint32_t>{ begin, end } };
	}
} // namespace vuk
