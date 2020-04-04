#include "vush.hpp"
#include <algorithm>

// haha, computer cache goes bzzzz
std::unordered_map<std::string, std::unordered_map<std::string, rule>> rules;

std::regex parse_parameters_regex(R"(\s*(?:(\w+)\s*::)?\s*(\w+)\s*(\w+))");
std::regex find_struct(R"(\s*struct\s*(\w+)\s*\{([\s\S]*?)\};)");
std::regex parse_struct_members(R"(\s*(?:layout\((.*)\))?\s*(?:(\w+)::)?\s*(\w+)\s*(\w+))");
std::regex pragma_regex(R"(#pragma\s*(\w+)?\s+(?:(\w+)\s*::)?\s*([\w\/]+)\s*:\s*(\S+))");
std::regex find_stages(R"((\w+)\s*(\w+?)\s*::\s*(\w+)\s*\((.+)\)(\s*\{[\s\S]+?\}))");
std::regex include_regex(R"(#include\s*(?:"\s*(\S+)\s*")|#include\s*(?:<\s*(\S+)\s*>))");

namespace {
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
}
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

std::vector<parameter_entry> parse_parameters(const std::string& src, std::unordered_map<std::string, std::vector<parameter_entry>>& parameters_per_scope) {
	std::vector<parameter_entry> params;
	auto words_begin = std::sregex_iterator(src.begin(), src.end(), parse_parameters_regex);
	auto words_end = std::sregex_iterator();

	for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
		std::smatch match = *i;
		std::string match_str = match.str();
		parameter_entry se;
		se.scope = match[1].matched ? match[1].str() : "Stage";
		se.type = match[2].str();
		se.name = match[3].str();
		parameters_per_scope[se.scope].push_back(se);
		params.push_back(std::move(se));
	}
	return params;
}

stage_entry::type to_stage(const std::string& i) {
	if (i == "vertex") return stage_entry::type::eVertex;
	if (i == "fragment") return stage_entry::type::eFragment;
	assert(0);
	return stage_entry::type::eVertex;
}

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

void parse_pragmas(const std::string& prefix, std::unordered_map<std::string, meta>& metadata) {
	auto words_begin = std::sregex_iterator(prefix.begin(), prefix.end(), pragma_regex);
	auto words_end = std::sregex_iterator();
	for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
		std::smatch match = *it;

		setting s;
		s.modifier = match[1].matched ? setting::to_modifier(match[1].str()) : setting::Mod::eNone;
		if (match[2].matched)
			s.aspect = match[2].str();
		s.name = match[3].str();
		s.value = match[4].str();

		if (s.aspect) {
			metadata[*s.aspect].settings.push_back(s);
		} else {
			metadata["_all"].settings.push_back(s);
		}
	}

}

void add_rules(json in_json) {
	for (auto& [scope, impls] : in_json.items()) {
		for (auto& [name, impl] : impls.items()) {
			rule r;
			r.is_unique = impl.value("unique", false);
			if (impl["stages"].is_null()) {
				r.stages = { stage_entry::type::eVertex,  stage_entry::type::eFragment }; // all stages
			} else {
				for (auto& s : impl["stages"]) {
					r.stages.push_back(to_stage(s.get<std::string>()));
				}
			}
			r.declaration_template = mustache::mustache(impl.value("declaration_template", ""));
			assert(r.declaration_template.is_valid());

			r.bind_template = mustache::mustache(impl.value("bind_template", ""));
			assert(r.bind_template.is_valid());

			r.location_count = mustache::mustache(impl.value("location_count", ""));
			assert(r.location_count.is_valid());

			r.binding_count = mustache::mustache(impl.value("binding_count", ""));
			assert(r.binding_count.is_valid());

			for (auto& [pname, parm] : impl["parameters"].items()) {
				r.parameters.push_back(rule::parameter{ pname, parm["value"].get<std::string>() });
			}

			rules[scope][name] = r;
		}
	}
}

std::unordered_map<std::string, rule>& find_ruleset(const std::string& scope_name) {
	auto it = rules.find(scope_name);
	if (it == rules.end())
		throw "Ruleset not found";
	return it->second;
}

void generate(const char* filename, stage_entry& se, const std::unordered_map<std::string, struct_entry>& structs, const std::unordered_map<std::string, meta>& metadata, const std::unordered_map<std::string, std::vector<parameter_entry>>& parameters_per_scope, std::unordered_map<std::string, uint32_t>& bindings, generate_result& gresult) {
	std::stringstream result;
	// assemble template hash
	result << "// file generated by vush compiler, from " << filename << '\n';
	// preamble
	result << "#version 460\n";
	result << "#pragma shader_stage(" << se.stage_as_string << ")\n";
	result << "#extension GL_GOOGLE_cpp_style_line_directive : require\n";
	result << "#extension GL_GOOGLE_include_directive : require\n";
	result << "\n";
	result << se.context;

	// emit out handling
	if (se.stage == stage_entry::type::eVertex) {
		result << "layout(location = 0) out " << se.return_type << " _out;\n";
	} else if (se.stage == stage_entry::type::eFragment) {
		auto& str = structs.at(se.return_type);
		size_t index = 0;
		for (auto& m : str.members) {
			result << "layout(location = " << index << ") out " << m.type << " _" << m.name << "_out;\n";
			index++;
		}
	}

	result << "\n";

	// emit scope var declarations
	size_t binding = 0;
	size_t location = 0;
	for (auto& [scope_name, params] : parameters_per_scope) {
		auto& ruleset = find_ruleset(scope_name);
		auto& rule = ruleset.begin()->second;

		for (auto& parm : rule.parameters) {
			result << "#define " << parm.name << " " << parm.value << "\n";
		}

		auto hash = se.to_hash(structs, parameters_per_scope, scope_name, (scope_name == "Stage" || scope_name == "Attribute") ? true : false);
		hash["binding"] = std::to_string(binding);
		hash["location"] = std::to_string(location);

		result << rule.declaration_template.render(hash);

		auto location_count = rule.location_count.render(hash).length();
		location += location_count;

		auto binding_count = rule.binding_count.render(hash).length();
		binding += binding_count;

	}
	result << '\n';

	// emit original function
	result << "#line " << se.signature_line_number << " \"" << filename << "\"\n";
	result << se.return_type << " " << se.aspect_name << "_" << se.stage_as_string << "(";
	for (auto i = 0; i < se.parameters.size(); i++) {
		auto& p = se.parameters[i];
		result << p.type << " " << p.name;
		if (i < se.parameters.size() - 1)
			result << ", ";
	}
	result << ")";
	result << se.body << '\n';

	result << "\n";

	// emit main function
	result << "void main() {\n";
	// apply bind rules
	for (auto& [scope_name, params] : parameters_per_scope) {
		auto& ruleset = find_ruleset(scope_name);
		auto& rule = ruleset.begin()->second;

		auto hash = se.to_hash(structs, parameters_per_scope, scope_name, true);
		result << rule.bind_template.render(hash);
	}

	if (se.stage != stage_entry::type::eFragment) {
		result << "\t_out = ";
	} else {
		result << "\t" << se.return_type << " _out = ";
	}
	result << se.aspect_name << "_" << se.stage_as_string << "(";
	for (auto i = 0; i < se.parameters.size(); i++) {
		auto& p = se.parameters[i];
		result << p.name;
		if (i < se.parameters.size() - 1)
			result << ", ";
	}
	result << ");\n";
	if (se.stage == stage_entry::type::eFragment) { // destructure the result into the out params
		auto& str = structs.at(se.return_type);
		for (auto& m : str.members) {
			result << "\t" << "_" << m.name << "_out = _out." << m.name << ";\n";
		}

	}
	result << "}";

	generate_result::per_aspect& pa = gresult.aspects[se.aspect_name];
	pa.shaders.emplace_back(generate_result::per_aspect::shader{ se.stage, result.str() });
	auto mit = metadata.find(se.aspect_name);
	pa.metadata = mit != metadata.end() ? mit->second : meta{};
	auto ait = metadata.find("_all");
	if (ait != metadata.end())
		pa.metadata.settings.insert(pa.metadata.settings.end(), ait->second.settings.begin(), ait->second.settings.end());
}

void parse_includes(const std::string& str, std::unordered_map<std::string, struct_entry>& structs, std::unordered_map<std::string, meta>& metadata) {
	auto words_begin = std::sregex_iterator(str.begin(), str.end(), include_regex);
	auto words_end = std::sregex_iterator();
	for (std::sregex_iterator it = words_begin; it != words_end; ++it) {
		std::smatch match = *it;
		bool local = match[1].matched;
		bool system = match[2].matched;
		std::string full_path = std::string("../../tests/") + (local ? match[1].str() : match[2].str()); // TODO
		auto contents = slurp(full_path);

		parse_includes(contents, structs, metadata);
	}

	parse_structs(str, structs);
	parse_pragmas(str, metadata);
}

void parse_context(const std::string& prefix, std::unordered_map<std::string, struct_entry>& structs, std::unordered_map<std::string, meta>& metadata) {
	parse_includes(prefix, structs, metadata);
}

generate_result parse_generate(const std::string& src, const char* filename) {
	generate_result gresult;

	std::unordered_map<std::string, struct_entry> structs;
	std::unordered_map<std::string, meta> metadata;
	std::unordered_map<std::string, std::vector<parameter_entry>> parameters_per_scope;
	std::vector<stage_entry> stages;
	std::string context = "";

	auto words_begin = std::sregex_iterator(src.begin(), src.end(), find_stages);
	auto words_end = std::sregex_iterator();
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

		se.parameters = parse_parameters(match[4].str(), parameters_per_scope);
		se.body = match[5].str();
		stages.push_back(std::move(se));

		// parse prefix for structs & metadata & includes
		parse_context(prefix, structs, metadata);
	}
	std::unordered_map<std::string, uint32_t> bindings;
	for(auto& se : stages){
		generate(filename, se, structs, metadata, parameters_per_scope, bindings, gresult);
	}

	for (auto& [aspect, res] : gresult.aspects) {
		auto& j = res.metadata_as_json;
		for (auto& s : res.metadata.settings) {
			j[s.name] = s.value;
		}
	}

	return gresult;
}

bool is_opaque_type(const std::string& t) {
	if (t == "sampler2D") return true;
	return false;
}

mustache::data stage_entry::to_hash(const std::unordered_map<std::string, struct_entry>& structs, const std::unordered_map<std::string, std::vector<parameter_entry>>& per_scope, const std::string& aspect, bool use) {
	mustache::data root;
	mustache::data params{ mustache::data::type::list };
	size_t index = 0;
	size_t variable_index = 0;
	size_t opaque_index = 0;
	if (use) {
		for (auto& p : parameters) {
			if (p.scope != aspect) continue;
			if (is_opaque_type(p.type)) continue;
			mustache::data d;
			d["scope"] = p.scope;
			d["variable_type"] = p.type;
			d["variable_name"] = p.name;

			auto it = structs.find(p.type);
			bool is_struct = it != structs.end();
			d["is_struct"] = is_struct;
			if (is_struct) {
				size_t member_index = 0;
				mustache::data members{ mustache::data::type::list };
				for (auto& m : it->second.members) {
					mustache::data memb;
					memb["type"] = m.type;
					memb["name"] = m.name;
					memb["index"] = std::to_string(index++);
					memb["member_index"] = std::to_string(member_index++);

					members.push_back(memb);
				}
				d.set("members", members);
			}
			d["variable_index"] = std::to_string(variable_index++);
			params.push_back(d);
		}
		root.set("variables", params);
		if (params.list_value().size() > 0)
			root.set("variables_exist", true);
		params = { mustache::data::type::list };
		for (auto& p : parameters) {
			if (p.scope != aspect) continue;
			if (!is_opaque_type(p.type)) continue;
			mustache::data d;
			d["scope"] = p.scope;
			d["variable_type"] = p.type;
			d["variable_name"] = p.name;

			d["variable_index"] = std::to_string(opaque_index++);
			params.push_back(d);
		}
		root.set("opaque_variables", params);
		if (params.list_value().size() > 0)
			root.set("opaque_variables_exist", true);
	} else {
		for (auto& [scope, s] : per_scope) {
			for (auto& p : s) {
				if (scope != aspect) continue;
				if (is_opaque_type(p.type)) continue;
				mustache::data d;
				d["scope"] = p.scope;
				d["variable_type"] = p.type;
				d["variable_name"] = p.name;

				auto it = structs.find(p.type);
				bool is_struct = it != structs.end();
				d["is_struct"] = is_struct;
				if (is_struct) {
					size_t member_index = 0;
					mustache::data members{ mustache::data::type::list };
					for (auto& m : it->second.members) {
						mustache::data memb;
						memb["type"] = m.type;
						memb["name"] = m.name;
						memb["index"] = std::to_string(index++);
						memb["member_index"] = std::to_string(member_index++);

						members.push_back(memb);
					}
					d.set("members", members);
				}
				d["variable_index"] = std::to_string(variable_index++);
				params.push_back(d);
			}
		}
		root.set("variables", params);
		if (params.list_value().size() > 0)
			root.set("variables_exist", true);
		params = { mustache::data::type::list };
		for (auto& [scope, s] : per_scope) {
			if (scope != aspect) continue;
			for (auto& p : s) {
				if (!is_opaque_type(p.type)) continue;
				mustache::data d;
				d["scope"] = p.scope;
				d["variable_type"] = p.type;
				d["variable_name"] = p.name;

				d["variable_index"] = std::to_string(opaque_index++);
				params.push_back(d);
			}
		}
		root.set("opaque_variables", params);
		if (params.list_value().size() > 0)
			root.set("opaque_variables_exist", true);
	}
	return root;
}

setting::Mod setting::to_modifier(const std::string& s) {
	if (s == "force") {
		return Mod::eForce;
	} else {
		throw "Unknown modifier";
	}
}
