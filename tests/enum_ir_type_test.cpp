#include "TestContext.hpp"
#include "vuk/ir/IR.hpp"
#include "vuk/ir/IRCppTypes.hpp"
#include <doctest/doctest.h>

using namespace vuk;

enum class TestEnum : uint32_t {
	Value1 = 1,
	Value2 = 2,
	Value3 = 3
};

// Provide format_as for TestEnum
std::string format_as(TestEnum e) {
	switch (e) {
	case TestEnum::Value1:
		return "Value1";
	case TestEnum::Value2:
		return "Value2";
	case TestEnum::Value3:
		return "Value3";
	default:
		return "Unknown";
	}
}

TEST_CASE("enum_ir_type_creation") {
	// Test that enum types are created with correct tag from typeid
	auto enum_ty = to_IR_type<TestEnum>();
	
	CHECK(enum_ty->kind == Type::ENUM_TY);
	CHECK(enum_ty->size == sizeof(TestEnum));
	CHECK(enum_ty->enumt.tag == typeid(TestEnum).hash_code());
	CHECK(enum_ty->enumt.format_to != nullptr);
}

TEST_CASE("enum_ir_type_debug_info") {
	auto enum_ty = to_IR_type<TestEnum>();
	
	// Check that debug info contains the type name
	CHECK_FALSE(enum_ty->debug_info.name.empty());
	// The name should contain "TestEnum"
	CHECK(enum_ty->debug_info.name.find("TestEnum") != std::string::npos);
}

TEST_CASE("enum_ir_type_formatting") {
	auto enum_ty = to_IR_type<TestEnum>();
	
	TestEnum test_value = TestEnum::Value2;
	std::string formatted;
	
	if (enum_ty->enumt.format_to) {
		enum_ty->enumt.format_to(&test_value, formatted);
		CHECK(formatted == "Value2");
	}
}

TEST_CASE("enum_ir_type_to_string") {
	auto enum_ty = to_IR_type<TestEnum>();
	
	auto type_str = Type::to_string(enum_ty.get());
	// Should use debug name if available
	if (!enum_ty->debug_info.name.empty()) {
		CHECK(type_str == enum_ty->debug_info.name);
	} else {
		CHECK(type_str.find("enum:") != std::string::npos);
	}
}

TEST_CASE("enum_ir_type_hash") {
	auto enum_ty1 = to_IR_type<TestEnum>();
	auto enum_ty2 = to_IR_type<TestEnum>();
	
	// Same enum type should produce same hash
	CHECK(Type::hash(enum_ty1.get()) == Type::hash(enum_ty2.get()));
}

// Test with a namespaced enum
namespace test_namespace {
	enum class NamespacedEnum {
		OptionA,
		OptionB
	};
}

TEST_CASE("namespaced_enum_debug_info") {
	auto enum_ty = to_IR_type<test_namespace::NamespacedEnum>();
	
	// Check that debug info contains the full namespaced name
	CHECK_FALSE(enum_ty->debug_info.name.empty());
	// The name should contain both namespace and enum name
	CHECK(enum_ty->debug_info.name.find("NamespacedEnum") != std::string::npos);
}

TEST_CASE("enum_value_type_creation") {
	auto module = std::make_shared<IRModule>();
	auto enum_ty = to_IR_type<TestEnum>();
	
	// Create an enum value type
	auto enum_value_ty = module->types.make_enum_value_ty(enum_ty, static_cast<uint64_t>(TestEnum::Value2));
	
	CHECK(enum_value_ty->kind == Type::ENUM_VALUE_TY);
	CHECK(enum_value_ty->size == sizeof(TestEnum));
	CHECK(enum_value_ty->enum_value.value == static_cast<uint64_t>(TestEnum::Value2));
	CHECK(enum_value_ty->enum_value.enum_type->get() == enum_ty.get());
}

TEST_CASE("enum_value_constant") {
	auto module = std::make_shared<IRModule>();
	current_module = module;
	
	auto enum_ty = to_IR_type<TestEnum>();
	
	// Create an enum constant using the helper function
	auto enum_const = module->make_enum_constant(enum_ty, TestEnum::Value3);
	
	CHECK(enum_const.type()->kind == Type::ENUM_VALUE_TY);
	CHECK(enum_const.type()->enum_value.value == static_cast<uint64_t>(TestEnum::Value3));
	CHECK(enum_const.node->kind == Node::CONSTANT);
	
	current_module.reset();
}

TEST_CASE("enum_value_type_to_string") {
	auto module = std::make_shared<IRModule>();
	auto enum_ty = to_IR_type<TestEnum>();
	
	auto enum_value_ty = module->types.make_enum_value_ty(enum_ty, static_cast<uint64_t>(TestEnum::Value1));
	
	auto type_str = Type::to_string(enum_value_ty.get());
	// Should format as "EnumType::Value"
	CHECK(type_str.find("::") != std::string::npos);
	CHECK(type_str.find("Value1") != std::string::npos);
}

TEST_CASE("enum_value_type_hash") {
	auto module = std::make_shared<IRModule>();
	auto enum_ty = to_IR_type<TestEnum>();
	
	// Create two enum value types with the same value
	auto enum_value_ty1 = module->types.make_enum_value_ty(enum_ty, static_cast<uint64_t>(TestEnum::Value2));
	auto enum_value_ty2 = module->types.make_enum_value_ty(enum_ty, static_cast<uint64_t>(TestEnum::Value2));
	
	// Same enum type and value should produce same hash
	CHECK(Type::hash(enum_value_ty1.get()) == Type::hash(enum_value_ty2.get()));
	
	// Different values should produce different hashes
	auto enum_value_ty3 = module->types.make_enum_value_ty(enum_ty, static_cast<uint64_t>(TestEnum::Value3));
	CHECK(Type::hash(enum_value_ty1.get()) != Type::hash(enum_value_ty3.get()));
}
