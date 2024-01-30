#include "vuk/Exception.hpp"
#include "vuk/Result.hpp"
#include "vuk/ShaderSource.hpp"

#include <filesystem>
#include <fmt/format.h>
#include <fstream>

extern "C" {
namespace shady {
#include "shady/driver.h"
}
}

#ifdef _MSC_VER
#define popen  _popen
#define pclose _pclose
#endif

inline std::string read_entire_file(const std::string& path) {
	std::ostringstream buf;
	std::ifstream input(path.c_str());
	assert(input);
	buf << input.rdbuf();
	return buf.str();
}

namespace vuk {
	Result<std::vector<uint32_t>> compile_c(const ShaderModuleCreateInfo& cinfo, uint32_t shader_compiler_target_version) {
		shady::DriverConfig driver_config = shady::default_driver_config();
		shady::CompilerConfig compiler_config = shady::default_compiler_config();
		compiler_config.specialization.entry_point = "main";
		//compiler_config.specialization.execution_model = shady::ExecutionModel::EmFragment;

		std::filesystem::path f(cinfo.filename);
		auto tmp_filename = std::filesystem::temp_directory_path() / f.filename();

		auto arg_string =
		    fmt::format("clang -c -emit-llvm -S -g -O0 -ffreestanding -Wno-main-return-type -Xclang -fpreserve-vec3-type --target=spir64-unknown-unknown "
		                "-isystem\"" VUK_VCC_INCLUDE_DIR "\" -D__SHADY__=1 -o {} {}",
		                tmp_filename.generic_string(),
		                cinfo.filename);
		FILE* stream = popen(arg_string.c_str(), "r");
		int clang_returned = pclose(stream);
		fmt::println("Clang returned {}", clang_returned);

		auto llvm_ir = read_entire_file(tmp_filename.generic_string());
		size_t len = llvm_ir.size();

		shady::ArenaConfig aconfig = shady::default_arena_config();
		aconfig.untyped_ptrs = true; 
		shady::IrArena* arena = new_ir_arena(aconfig);
		shady::Module* mod = new_module(arena, cinfo.filename.c_str());
		shady::driver_load_source_file(shady::SrcLLVM, len, llvm_ir.c_str(), mod);
		//shady::driver_compile(&driver_config, mod);
		shady::CompilationResult result = run_compiler_passes(&compiler_config, &mod);
		std::vector<uint32_t> spirv;
		size_t output_size;
		char* output_buffer;
		shady::emit_spirv(&compiler_config, mod, &output_size, &output_buffer, NULL);
		spirv.resize(output_size / 4);
		memcpy(&spirv[0], output_buffer, output_size);
		free(output_buffer);
		shady::destroy_ir_arena(arena);
		shady::destroy_driver_config(&driver_config);
		return { expected_value, spirv };
	}
} // namespace vuk