#ifdef VUK_BUILD_TESTS
#define DOCTEST_CONFIG_IMPLEMENT
#endif
#include <doctest/doctest.h>

#ifdef VUK_TEST_RUNNER
#include "TestContext.hpp"
namespace vuk {
	TestContext test_context;
}
int main(int argc, char** argv) {
	return doctest::Context(argc, argv).run();
}
#endif