#include <thread>

#include "TestContext.hpp"
#include <doctest/doctest.h>
#include "vuk/RadixTree.hpp"


using namespace vuk;

TEST_CASE("radix map insert 1") {
	RadixTree<int> foo;
	foo.insert(0x1, 1, 1);
	foo.insert(0x2, 1, 2);

	auto p = foo.find(0x1);
	CHECK(p);
	CHECK(*p == 1);
	p = foo.find(0x2);
	CHECK(*p == 2);
}

TEST_CASE("radix map insert 2") {
	RadixTree<int> foo;
	foo.insert(0x2, 2, 2);

	auto p = foo.find(0x1);
	CHECK(!p);
	p = foo.find(0x2);
	CHECK(*p == 2);
	p = foo.find(0x3);
	CHECK(*p == 2);
}

TEST_CASE("radix map insert 3") {
	RadixTree<int> foo;
	foo.insert(0x2, 1, 2);

	auto p = foo.find(0x2);
	CHECK(*p == 2);
	p = foo.find(0x1);
	CHECK(!p);
	p = foo.find(0x3);
	CHECK(!p);
}

TEST_CASE("radix map insert 4") {
	RadixTree<int> foo;
	foo.insert(0x2, 2, 2);
	foo.insert(0x1, 1, 1);
	auto p = foo.find(0x1);
	CHECK(*p == 1);
	p = foo.find(0x2);
	CHECK(*p == 2);
}

TEST_CASE("radix map insert unaligned") {
	RadixTree<int> foo;
	int size = 4;
	int base = 0x3;

	foo.insert_unaligned(base, size, 2);
	for (int i = 0; i < size; i++) {
		auto p = foo.find(base + i);
		CHECK(p);
		CHECK(*p == 2);
	}
}

#include <random>
TEST_CASE("radix map insert unaligned single") {

	std::mt19937 rng(4); // scientifically chosen
	const size_t MAX_BASE = 1024 * 1024;
	const size_t MAX_SIZE = 2048;
	std::uniform_int_distribution<std::mt19937::result_type> base_dist(1, MAX_BASE);
	std::uniform_int_distribution<std::mt19937::result_type> size_dist(1, MAX_SIZE);

	for (int j = 0; j < 100; j++) {
		int base = base_dist(rng);
		int size = size_dist(rng);
		RadixTree<int> foo;
		foo.insert_unaligned(base, size, 2);
		for (int i = 0; i < base; i++) {
			auto p = foo.find(i);
			CHECK(!p);
		}
		for (int i = 0; i < size; i++) {
			auto p = foo.find(base + i);
			CHECK(p);
			CHECK(*p == 2);
		}
		for (int i = base+size; i < MAX_BASE + MAX_SIZE; i++) {
			auto p = foo.find(i);
			CHECK(!p);
		}
	}
}

TEST_CASE("radix map insert unaligned multi") {
	std::mt19937 rng(4); // scientifically chosen
	const size_t MAX_BASE = 1024 * 1024;
	const size_t MAX_SIZE = 16;
	std::uniform_int_distribution<std::mt19937::result_type> base_dist(1, MAX_BASE);
	std::uniform_int_distribution<std::mt19937::result_type> size_dist(1, MAX_SIZE);

	std::unordered_map<size_t, size_t> to_find;
	RadixTree<int> foo;

	int base = 10;
	for (int j = 0; j < 100; j++) {
		base += size_dist(rng);
		int size = size_dist(rng);
		foo.insert_unaligned(base, size, size);
		for (int k = base; k < (base + size); k++) {
			to_find[k] = size;
		}
		base += size;
	}

	for (int i = 0; i < base + MAX_SIZE; i++) {
		auto p = foo.find(i);
		auto it = to_find.find(i);
		if (it != to_find.end()) {
			CHECK(p);
			CHECK(*p == it->second);
		} else {
			CHECK(!p);
		}
	}
}

TEST_CASE("radix map erase 1") {
	RadixTree<int> foo;
	foo.insert(0x2, 1, 2);

	foo.erase(0x2);
	auto p = foo.find(0x2);
	CHECK(p == nullptr);
}

TEST_CASE("radix map insert-erase unaligned single") {
	std::mt19937 rng(4); // scientifically chosen
	const size_t MAX_BASE = 1024 * 1024;
	const size_t MAX_SIZE = 2048;
	std::uniform_int_distribution<std::mt19937::result_type> base_dist(1, MAX_BASE);
	std::uniform_int_distribution<std::mt19937::result_type> size_dist(1, MAX_SIZE);

	for (int j = 0; j < 100; j++) {
		int base = base_dist(rng);
		int size = size_dist(rng);
		RadixTree<int> foo;
		foo.insert_unaligned(base, size, 2);
		foo.erase_unaligned(base, size);
		for (int i = 0; i < MAX_BASE + MAX_SIZE; i++) {
			auto p = foo.find(i);
			CHECK(!p);
		}
	}
}

TEST_CASE("radix map insert-erase unaligned multi") {
	std::mt19937 rng(4); // scientifically chosen
	const size_t MAX_BASE = 1024 * 1024;
	const size_t MAX_SIZE = 16;
	std::uniform_int_distribution<std::mt19937::result_type> base_dist(1, MAX_BASE);
	std::uniform_int_distribution<std::mt19937::result_type> size_dist(1, MAX_SIZE);

	std::unordered_map<size_t, std::pair<size_t, size_t>> to_find;
	RadixTree<std::pair<size_t, size_t>> foo;

	int base = 10;
	for (int j = 0; j < 100; j++) {
		base += size_dist(rng);
		int size = size_dist(rng);
		foo.insert_unaligned(base, size, {base, size});
		for (int k = base; k < (base + size); k++) {
			to_find[k] = { base, size };
		}
		base += size;
	}

	base = 10;

	for (int j = 0; j < 20; j++) {
		base += 5*size_dist(rng);
		auto p = foo.find(base);
		if (p) {
			auto& [base, size] = *p;
			for (int k = base; k < (base + size); k++) {
				to_find.erase(k);
			}
			foo.erase_unaligned(base, size);
		}
	}

	for (int i = 0; i < base + MAX_SIZE; i++) {
		auto p = foo.find(i);
		auto it = to_find.find(i);
		if (it != to_find.end()) {
			CHECK(p);
			CHECK(*p == it->second);
		} else {
			CHECK(!p);
		}
	}
}