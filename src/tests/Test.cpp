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

struct DT : public doctest::IReporter {
	// caching pointers/references to objects of these types - safe to do
	std::ostream& stdout_stream;
	const doctest::ContextOptions& opt;
	const doctest::TestCaseData* tc;
	std::mutex mutex;

	// constructor has to accept the ContextOptions by ref as a single argument
	DT(const doctest::ContextOptions& in) : stdout_stream(*in.cout), opt(in), tc(nullptr) {}

	void report_query(const doctest::QueryData& /*in*/) override {}

	void test_run_start() override {}

	void test_run_end(const doctest::TestRunStats& /*in*/) override {}

	void test_case_start(const doctest::TestCaseData& in) override {
		tc = &in;
		vuk::test_context.start(in.m_name);
	}

	void test_case_reenter(const doctest::TestCaseData& /*in*/) override {}

	void test_case_end(const doctest::CurrentTestCaseStats& /*in*/) override {
		vuk::test_context.finish();
	}

	void test_case_exception(const doctest::TestCaseException& /*in*/) override {}

	void subcase_start(const doctest::SubcaseSignature& /*in*/) override {}

	void subcase_end() override {}

	void log_assert(const doctest::AssertData& in) override {}

	void log_message(const doctest::MessageData& /*in*/) override {}

	void test_case_skipped(const doctest::TestCaseData& /*in*/) override {}
};

REGISTER_LISTENER("my_listener", 1, DT);
#endif