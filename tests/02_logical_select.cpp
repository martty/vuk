#include "TestContext.hpp"
#include "vuk/ir/IRCppTypes.hpp"
#include "vuk/ir/IRPass.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <algorithm>
#include <cstdint>
#include <doctest/doctest.h>
#include <string_view>

using namespace vuk;

// =================================================================
// Tests for Value<T> Logical and Comparison Operators
// =================================================================

TEST_CASE("boolean_constants") {
	Value<bool> true_val(true);
	Value<bool> false_val(false);

	CHECK(true_val.get_head().type()->is_boolean());
	CHECK(false_val.get_head().type()->is_boolean());

	CHECK(*true_val == true);
	CHECK(*false_val == false);
}

TEST_CASE("value_bool_logical_and_or") {
	Value<bool> true_val(true);
	Value<bool> false_val(false);

	// true && true = true
	auto and_true_true = true_val && true_val;
	CHECK(*and_true_true == true);

	// true && false = false
	auto and_true_false = true_val && false_val;
	CHECK(*and_true_false == false);

	// false && false = false
	auto and_false_false = false_val && false_val;
	CHECK(*and_false_false == false);

	// true || true = true
	auto or_true_true = true_val || true_val;
	CHECK(*or_true_true == true);

	// true || false = true
	auto or_true_false = true_val || false_val;
	CHECK(*or_true_false == true);

	// false || false = false
	auto or_false_false = false_val || false_val;
	CHECK(*or_false_false == false);
}

TEST_CASE("value_comparisons") {
	Value<uint32_t> a(42);
	Value<uint32_t> b(99);
	Value<uint32_t> c(42);

	// Test equality
	auto eq_true = a == c;
	CHECK(*eq_true == true);

	auto eq_false = a == b;
	CHECK(*eq_false == false);

	// Test inequality
	auto ne_true = a != b;
	CHECK(*ne_true == true);

	auto ne_false = a != c;
	CHECK(*ne_false == false);

	// Test less than
	auto lt_true = a < b;
	CHECK(*lt_true == true);

	auto lt_false = b < a;
	CHECK(*lt_false == false);

	// Test less than or equal
	auto le_true = a <= c;
	CHECK(*le_true == true);

	auto le_false = b <= a;
	CHECK(*le_false == false);

	// Test greater than
	auto gt_true = b > a;
	CHECK(*gt_true == true);

	auto gt_false = a > b;
	CHECK(*gt_false == false);

	// Test greater than or equal
	auto ge_true = a >= c;
	CHECK(*ge_true == true);

	auto ge_false = a >= b;
	CHECK(*ge_false == false);
}

TEST_CASE("value_select_basic") {
	Value<bool> condition_true(true);
	Value<bool> condition_false(false);
	Value<uint32_t> val_a(100);
	Value<uint32_t> val_b(200);

	auto result_true = select(condition_true, val_a, val_b);
	CHECK(*result_true == 100);

	auto result_false = select(condition_false, val_a, val_b);
	CHECK(*result_false == 200);
}

TEST_CASE("value_select_with_comparison") {
	Value<uint32_t> a(10);
	Value<uint32_t> b(20);
	Value<uint32_t> val_true(100);
	Value<uint32_t> val_false(200);

	auto result = select(a < b, val_true, val_false);
	CHECK(*result == 100); // 10 < 20 is true, so select val_true
}

TEST_CASE("value_select_64bit") {
	Value<bool> condition(true);
	Value<uint64_t> val_a(0x123456789ABCDEF0ULL);
	Value<uint64_t> val_b(0xFEDCBA9876543210ULL);

	auto result = select(condition, val_a, val_b);
	CHECK(*result == 0x123456789ABCDEF0ULL);
}

TEST_CASE("value_chained_comparisons") {
	// Test chaining comparisons with logical operators
	Value<uint32_t> x(15);
	Value<uint32_t> min_val(10);
	Value<uint32_t> max_val(20);

	// Check if x is in range [min_val, max_val]
	auto in_range = (x >= min_val) && (x <= max_val);
	CHECK(*in_range == true);

	// Test with out of range value
	Value<uint32_t> y(25);
	auto out_of_range = (y >= min_val) && (y <= max_val);
	CHECK(*out_of_range == false);
}

TEST_CASE("value_min_max_implementation") {
	// Implement max using select and comparison operators
	Value<uint32_t> a(42);
	Value<uint32_t> b(99);

	// max(a, b) = select(a > b, a, b)
	auto max_val = select(a > b, a, b);
	CHECK(*max_val == 99);

	// Test with swapped values
	auto max_val2 = select(b > a, b, a);
	CHECK(*max_val2 == 99);

	// min(a, b) = select(a < b, a, b)
	auto min_val = select(a < b, a, b);
	CHECK(*min_val == 42);

	// Test with swapped values
	auto min_val2 = select(b < a, b, a);
	CHECK(*min_val2 == 42);
}

TEST_CASE("value_clamp_implementation") {
	// Implement clamp using select and comparison operators
	// clamp(x, min, max) = select(x < min, min, select(x > max, max, x))
	Value<uint32_t> x(150);
	Value<uint32_t> min_val(0);
	Value<uint32_t> max_val(100);

	auto inner = select(x > max_val, max_val, x);
	auto result = select(x < min_val, min_val, inner);
	CHECK(*result == 100); // Clamped to max

	// Test clamping to min
	Value<uint32_t> y(0); // Value is already at min, but test with negative would clamp
	auto result2 = select(y < min_val, min_val, select(y > max_val, max_val, y));
	CHECK(*result2 == 0);

	// Test value in range
	Value<uint32_t> z(50);
	auto result3 = select(z < min_val, min_val, select(z > max_val, max_val, z));
	CHECK(*result3 == 50);
}

TEST_CASE("value_nested_select") {
	// Test nested select: if (a < 10) then 1 else (if (a < 20) then 2 else 3)
	Value<uint32_t> a(15);
	Value<uint32_t> ten(10);
	Value<uint32_t> twenty(20);
	Value<uint32_t> val1(1);
	Value<uint32_t> val2(2);
	Value<uint32_t> val3(3);

	auto inner_select = select(a < twenty, val2, val3);
	auto outer_select = select(a < ten, val1, inner_select);
	CHECK(*outer_select == 2); // a >= 10 and a < 20, so select val2

	// Test with a < 10
	Value<uint32_t> b(5);
	auto result2 = select(b < ten, val1, select(b < twenty, val2, val3));
	CHECK(*result2 == 1);

	// Test with a >= 20
	Value<uint32_t> c(25);
	auto result3 = select(c < ten, val1, select(c < twenty, val2, val3));
	CHECK(*result3 == 3);
}