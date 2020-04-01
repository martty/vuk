#include <string>
#include <fstream>
#include <sstream>
#include <regex>
#include <cassert>
#include <iostream>

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

static constexpr auto parse_parameter_text = R"(\s*(?:(\w+)\s*::)?\s*(\w+)\s*(\w+))";
struct parameter_entry {
	std::string scope;
	std::string type;
	std::string name;
};

std::regex find_struct( R"(\s*struct\s*(\w+)\s*\{([\s\S]*?)\};)");
std::regex parse_struct_members(R"(\s*(?:layout\((.*)\))?\s*(?:(\w+)::)?\s*(\w+)\s*(\w+))");

#include <optional>

struct struct_entry {
	std::string name;
	struct member {
		std::optional<std::string> layout;
		std::optional<std::string> scope;
		std::string type;
		std::string name;
	};
	std::vector<member> members;
};

struct_entry parse_struct(std::string name, const std::string& body) {
	struct_entry str;
	str.name = name;

	auto words_begin = std::sregex_iterator(body.begin(), body.end(), parse_struct_members);
	auto words_end = std::sregex_iterator();
	for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
		std::smatch match = *it;
		struct_entry::member member;
		member.type = match[3].str();
		member.name = match[4].str();
		if (match[1].matched) {
			member.layout = match[1];
		}
		if (match[2].matched) {
			member.scope = match[2];
		}
		str.members.push_back(member);
	}
	return str;
}

#include <unordered_map>

std::vector<parameter_entry> parse_parameters(const std::string& src, std::unordered_map<std::string, std::vector<parameter_entry>>& parameters_per_scope) {
	std::vector<parameter_entry> params;
	std::regex _(parse_parameter_text, std::regex_constants::ECMAScript);
    auto words_begin = std::sregex_iterator(src.begin(), src.end(), _);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
		std::smatch match = *i;
		std::string match_str = match.str();
		parameter_entry se;
		se.scope = match[1].matched ? match[1].str() : "Stage";
		se.type = match[2].str();
		se.name = match[3].str();
		std::cout << "   " << se.scope << "::" << se.type << " " << se.name << '\n';
		parameters_per_scope[se.scope].push_back(se);
		params.push_back(std::move(se));
	}
	return params;
}

static constexpr auto find_stages_text = R"((\w+)\s*(\w+?)\s*::\s*(\w+)\s*\((.+)\)(\s*\{[\s\S]+?\}))";
struct stage_entry {
	std::string context;
	std::string return_type;
	std::string aspect_name;
	enum class type {
		eVertex, eTCS, eTES, eGeometry, eFragment, eCompute
	} stage;
	std::string stage_as_string;
	size_t signature_line_number;
	std::vector<parameter_entry> parameters;
	std::string body;
};

stage_entry::type to_stage(const std::string& i) {
	if (i == "vertex") return stage_entry::type::eVertex;
	if (i == "fragment") return stage_entry::type::eFragment;
	assert(0);
	return stage_entry::type::eVertex;
}

#include <algorithm>

void parse_structs(const std::string& prefix, std::unordered_map<std::string, struct_entry>& structs) {
	auto words_begin = std::sregex_iterator(prefix.begin(), prefix.end(), find_struct);
	auto words_end = std::sregex_iterator();
	for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
		std::smatch match = *it;
		std::string name = match[1].str();
		std::string body = match[2].str();

		structs[name] = parse_struct(name, body);
	}
}

int main(int argc, char** argv) {
	if (argc < 2) return 0;
	for (size_t i = 1; i < argc; i++) {
		auto src = slurp(argv[i]);
		std::regex find_stages(find_stages_text, std::regex_constants::ECMAScript);
		auto words_begin = std::sregex_iterator(src.begin(), src.end(), find_stages);
		auto words_end = std::sregex_iterator();

		std::cout << argv[i] << ": Found "
			<< std::distance(words_begin, words_end)
			<< " stages\n";

		std::unordered_map<std::string, struct_entry> structs;
		std::string context = "";
		for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
			std::smatch match = *it;
			stage_entry se;
			auto pos = match.position(0);
			auto before = src.substr(0, pos);
			se.signature_line_number = std::count(before.begin(), before.end(), '\n') + 1;
			auto prefix = match.prefix().str();
			context += prefix;
			se.context = context;
			se.return_type = match[1].str();
			se.aspect_name = match[2].str();
			se.stage = to_stage(match[3].str());
			se.stage_as_string = match[3].str();
			std::cout << "  " << se.stage_as_string;

			std::unordered_map<std::string, std::vector<parameter_entry>> parameters_per_scope;
			se.parameters = parse_parameters(match[4].str(), parameters_per_scope);
			se.body = match[5].str();

			// parse prefix for structs
			parse_structs(prefix, structs);

			std::cout << "//////////\n";
			// preamble
			std::cout << "#version 460\n";
			std::cout << "#pragma stage(" << se.stage_as_string << ")\n";
			std::cout << "#extension GL_GOOGLE_cpp_style_line_directive : require\n";
			std::cout << "\n";
			std::cout << se.context;

			// emit out handling
			if (se.stage == stage_entry::type::eVertex) {
				std::cout << "layout(location = 0) out " << se.return_type << " _out;\n";
			} else if (se.stage == stage_entry::type::eFragment) {
				auto& str = structs.at(se.return_type);
				size_t index = 0;
				for (auto& m : str.members) {
					std::cout << "layout(location = " << index << ") out " << m.type << " _" << m.name << "_out;\n";
					index++;
				}
			}

			std::cout << "\n";

			// emit scope var declarations
			for (auto& [scope_name, params] : parameters_per_scope) {
				if (scope_name == "Attribute") {
					size_t attribute_index = 0;
					for (auto& att : params) {
						auto& str = structs.at(att.type);
						for (auto& m : str.members) {
							std::cout << "layout(location = " << attribute_index << ") in " << m.type << " ";
							std::cout << "_" << att.type << "_" << m.name << ";\n";
							attribute_index++;
						}
					}
				} else if (scope_name == "Stage") {
					assert(params.size() == 1);
					std::cout << "layout(location = 0) in " << params[0].type << " _in;\n";
				}
			}
			std::cout << '\n';

			// emit original function
			std::cout << "#line " << se.signature_line_number << " \"" << argv[i] << "\"\n";
			std::cout << se.return_type << " " << se.aspect_name << "_" << se.stage_as_string << "(";
			for (auto i = 0; i < se.parameters.size(); i++) {
				auto& p = se.parameters[i];
				std::cout << p.type << " " << p.name;
				if (i < se.parameters.size() - 1)
					std::cout << ", ";
			}
			std::cout << ")";
			std::cout << se.body << '\n';

			std::cout << "\n";

			// emit main function
			std::cout << "void main() {\n";
			
			for (auto& [scope_name, params] : parameters_per_scope) {
				for (auto& att : params) {
					if (scope_name == "Attribute") {
						std::cout << "\t" << att.type << " " << att.name << ";\n";
						auto& str = structs.at(att.type);
						for (auto& m : str.members) {
							std::cout << "\t" << att.name << "." << m.name << " = " << "_" << att.type << "_" << m.name << ";\n";
						}
					} else if (scope_name == "Stage") {

					}
				}
			}
			
			if (se.stage != stage_entry::type::eFragment) {
				std::cout << "\t_out = ";
			} else {
				std::cout << "\t " << se.return_type << " _out = ";
			}
			std::cout << se.aspect_name << "_" << se.stage_as_string << "(";
			for (auto i = 0; i < se.parameters.size(); i++) {
				auto& p = se.parameters[i];
				std::cout << p.name;
				if (i < se.parameters.size() - 1)
					std::cout << ", ";
			}
			std::cout << ");\n";
			if (se.stage == stage_entry::type::eFragment) { // destructure the result into the out params
				auto& str = structs.at(se.return_type);
				for (auto& m : str.members) {
					std::cout << "\t" << "_" << m.name << "_out = _out." << m.name << ";\n";
				}

			}
			std::cout << "}";

			std::cout << "\n/////////////////\n";
		}
	}
}
