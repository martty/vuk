#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include "vuk/Partials.hpp"
#include <doctest/doctest.h>

using namespace vuk;

TEST_CASE("error: unattached resource") {
	REQUIRE(test_context.prepare());

	std::shared_ptr<RenderGraph> rg = std::make_shared<RenderGraph>("uatt");
	rg->add_pass({
	    .resources = { "nonexistent_image"_image >> vuk::eColorWrite },
	});

	Compiler compiler;
	REQUIRE_THROWS(compiler.compile(std::span{ &rg, 1 }, {}));
}

TEST_CASE("error: cbuf references unknown resource") {
	REQUIRE(test_context.prepare());

	std::shared_ptr<RenderGraph> rg = std::make_shared<RenderGraph>("uatt");
	rg->add_pass({ .execute = [](vuk::CommandBuffer& cbuf) {
		cbuf.bind_buffer(0, 0, "foo");
	} });

	Compiler compiler;
	auto ex = compiler.link(std::span{ &rg, 1 }, {});
	REQUIRE((bool)ex);
	REQUIRE_THROWS(ex->execute(*test_context.allocator, {}));
}