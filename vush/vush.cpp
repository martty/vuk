#include "vush.hpp"


int main(int argc, char** argv) {
	add_rules(json::parse(slurp("../../vush/builtin_cfg.json")));

	if (argc < 2) return 0;
	for (size_t i = 1; i < argc; i++) {
		auto src = slurp(argv[i]);
		auto filename = argv[i];

		auto result = parse_generate(src, filename);

	}
}
