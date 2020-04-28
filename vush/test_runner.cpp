#include "vush.hpp"

#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch2/catch.hpp"
#include <filesystem>
namespace fs = std::filesystem;
using namespace vush;

std::string slurp(const std::string& path) {
	std::ostringstream buf;
	std::ifstream input(path.c_str());
	buf << input.rdbuf();
	return buf.str();
}

void burp(const std::string& in, const std::string& path) {
	std::ofstream output(path.c_str(), std::ios::trunc);
	if (!output.is_open()) {
	}
	output << in;
	output.close();
}

std::string stage_to_extension(vush::stage_entry::type as) {
	switch (as) {
	case stage_entry::type::eVertex: return "vert";
	case stage_entry::type::eFragment: return "frag";
	default: assert(0); return "";
	}
}

void run_file(const std::string& src_file) {
	auto src = slurp(src_file);
	auto gen = vush::parse_generate(src, src_file.c_str());
	size_t checks = 0;
	for (const auto& [aspect, pa] : gen.aspects) {
		for (auto& ps : pa.shaders) {
			auto control_file = src_file + "." + aspect + "." + stage_to_extension(ps.stage);
			if (fs::exists(fs::path(control_file))) {
				auto control = slurp(control_file);

				REQUIRE(ps.source == control);
				checks++;
			}
		}
		auto control_file = src_file + "." + aspect + ".meta.json";
		auto meta_dump = pa.metadata_as_json.dump();
		if (fs::exists(fs::path(control_file))) {
			auto control = slurp(control_file);
			
			REQUIRE(pa.metadata_as_json == json::parse(control));
			checks++;
		}
	}
	REQUIRE(checks > 0);

}
/*
TEST_CASE("basic", "[basic]") {
	add_rules(json::parse(slurp("../../vush/builtin_cfg.json")));

	run_file("../../tests/basic.vush");
}

TEST_CASE("aspect", "[basic]") {
	add_rules(json::parse(slurp("../../vush/builtin_cfg.json")));

	run_file("../../tests/aspect.vush");
}

TEST_CASE("bindless", "[bindless]") {
	add_rules(json::parse(slurp("../../vush/builtin_cfg.json")));

	run_file("../../tests/bindless.vush");
}

TEST_CASE("pipeline_stage", "[pragmas]") {
	add_rules(json::parse(slurp("../../vush/builtin_cfg.json")));

	run_file("../../tests/pipeline_state.vush");
}

TEST_CASE("sampling", "[textures]") {
	add_rules(json::parse(slurp("../../vush/builtin_cfg.json")));

	run_file("../../tests/sampling.vush");
}*/

TEST_CASE("probing", "[introspection]") {
	add_rules(json::parse(slurp("../../vush/builtin_cfg.json")));

	run_file("../../tests/probing_param.vush");
}