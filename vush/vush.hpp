#include <string>
#include <fstream>
#include <sstream>
#include <regex>
#include <cassert>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <mustache.hpp>
using namespace kainjow;
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace vush {
    struct parameter_entry {
        std::string scope;
        std::string type;
        std::string name;
    };

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

    struct_entry parse_struct(std::string name, const std::string& body);

    std::vector<parameter_entry> parse_parameters(const std::string& src, std::unordered_map<std::string, std::vector<parameter_entry>>& parameters_per_scope);

    struct probe_entry {
        uint32_t number;
        std::optional<std::string> type;
        std::string name;
        uint32_t line;
    };

    struct stage_entry {
        std::string context;
        std::string epilogue;
        std::string return_type;
        std::string aspect_name;
        enum type { eVertex = 0, eTCS = 1, eTES = 2, eGeometry = 3, eFragment = 4, eCompute = 5 } stage;
        std::string stage_as_string;
        size_t signature_line_number;
        std::vector<parameter_entry> parameters;
        mustache::data to_hash(const std::unordered_map<std::string, struct_entry>& structs,
                               const std::unordered_map<std::string, std::vector<parameter_entry>>& per_scope, const std::string& aspect, bool use);
        std::string body;
        std::vector<probe_entry> probes;
    };

    struct setting {
        enum class Mod { eNone, eForce } modifier;

        static Mod to_modifier(const std::string& s);

        std::optional<std::string> aspect;
        std::string name;
        std::string value;
    };

    struct meta {
        std::vector<setting> settings;
        std::vector<parameter_entry> parameters;
        std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> bindings;
    };

    stage_entry::type to_stage(const std::string& i);

    void parse_structs(const std::string& prefix, std::unordered_map<std::string, struct_entry>& structs);

    void parse_pragmas(const std::string& prefix, std::unordered_map<std::string, meta>& metadata);

    struct rule {
        bool is_unique;
        std::vector<stage_entry::type> stages;
        mustache::mustache declaration_template = mustache::mustache("");
        mustache::mustache bind_template = mustache::mustache("");
        mustache::mustache binding_count = mustache::mustache("");
        mustache::mustache location_count = mustache::mustache("");

        struct parameter {
            std::string name;
            std::string value;
        };
        std::vector<parameter> parameters;
    };

    void add_rules(json in_json);

    struct generate_result {
        struct per_aspect {
            struct shader {
                stage_entry::type stage;
                std::string source;
            };
            std::vector<shader> shaders;
            meta metadata;
            json metadata_as_json;
        };

        // by aspect
        std::unordered_map<std::string, per_aspect> aspects;
        std::unordered_map<std::string, struct_entry> structs;
    };

    std::unordered_map<std::string, rule>& find_ruleset(const std::string& scope_name);

    std::string preprocess(const std::string& str);

    void generate(const char* filename, stage_entry& se, const std::unordered_map<std::string, struct_entry>& structs,
                  const std::unordered_map<std::string, meta>& metadata,
                  const std::unordered_map<std::string, std::vector<parameter_entry>>& parameters_per_scope,
                  generate_result& gresult);

    void parse_includes(const std::string& str, std::unordered_map<std::string, struct_entry>& structs, std::unordered_map<std::string, meta>& metadata);

    void parse_context(const std::string& prefix, std::unordered_map<std::string, struct_entry>& structs, std::unordered_map<std::string, meta>& metadata);

    generate_result parse_generate(const std::string& src, const char* filename);
} // namespace vush
