#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <shaderc/shaderc.hpp>
#include <sstream>

namespace vuk {
	/// @brief This default includer will look in the current working directory of the app and relative to the includer file to resolve includes
	class ShadercDefaultIncluder : public shaderc::CompileOptions::IncluderInterface {
		struct IncludeData {
			std::string source;
			std::string content;
		};

		std::filesystem::path base_path = std::filesystem::current_path();
		static constexpr const char* runtime_include = 
#include "vuk/vsl/glsl/vuk_runtime.glsl"
			;

	public:
		// Handles shaderc_include_resolver_fn callbacks.
		shaderc_include_result* GetInclude(const char* requested_source, shaderc_include_type type, const char* requesting_source, size_t include_depth) override {
			auto data = new IncludeData;
			auto path = base_path / requested_source;
			auto alternative_path = std::filesystem::absolute(std::filesystem::path(requesting_source).remove_filename() / requested_source);
			std::ostringstream buf;
			if (type == shaderc_include_type::shaderc_include_type_standard && strcmp(requested_source, "runtime") == 0) {
				data->content = runtime_include;
				data->source = "/vuk/vsl/glsl/vuk_runtime.glsl";
			} else if (std::ifstream input(path); input) {
				buf << input.rdbuf();
				data->content = buf.str();
				data->source = path.string();
			} else if (input = std::ifstream(alternative_path); input) {
				buf << input.rdbuf();
				data->content = buf.str();
				data->source = alternative_path.string();
			} else {
				data->content = fmt::format("file could not be read (tried: {}; {})", path.string().c_str(), alternative_path.string().c_str());
			}

			shaderc_include_result* result = new shaderc_include_result;
			result->user_data = data;
			result->source_name = data->source.c_str();
			result->source_name_length = data->source.size();
			result->content = data->content.c_str();
			result->content_length = data->content.size();

			return result;
		}

		// Handles shaderc_include_result_release_fn callbacks.
		void ReleaseInclude(shaderc_include_result* data) override {
			delete static_cast<IncludeData*>(data->user_data);
			delete data;
		}
	};
} // namespace vuk