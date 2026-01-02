TEST_CASE("allocate view in IR") {
	auto buf0 = vuk::allocate<float>("jacob", { .memory_usage = MemoryUsage::eCPUonly, .size = 16 });

	clear(buf0, 0.f);

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 460
#pragma shader_stage(compute)
#include <runtime>

layout (push_constant) uniform data {
  REF(float) data_in;
};

layout (local_size_x = 1) in;

void main() {
  ARRAY(data_in)[gl_GlobalInvocationID.x] = (gl_GlobalInvocationID.x + 1);
}
)")));
	pass(4, 1, 1, buf0->ptr);
	auto res = *buf0.get(*test_context.allocator, test_context.compiler, { .dump_graph = true });
	auto test = { 1.f, 2.f, 3.f, 4.f };
	auto schpen = res.to_span();
	CHECK(schpen == std::span(test));
}

TEST_CASE("shader buffer access (view)") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_view<BufferLike<float>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	for (int i = 0; i < 4; i++) {
		foo[i] = (i + 1);
	}

	view<BufferLike<float>> v = foo.get();
	auto buf0 = vuk::acquire("b0", v, vuk::Access::eNone);

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 460
#pragma shader_stage(compute)
#include <runtime>

layout (std430, binding = 0) buffer coherent BufferIn {
	float[] data_in;
};

layout (local_size_x = 1) in;

void main() {
	data_in[gl_GlobalInvocationID.x] *= 2;
}
)")));
	pass(4, 1, 1, buf0);
	buf0.wait(*test_context.allocator, test_context.compiler);
	auto test = { 2.f, 4.f, 6.f, 8.f };
	auto schpen = std::span(&foo[0], 4);
	CHECK(schpen == std::span(test));
}

TEST_CASE("shader ptr access") {
	Allocator alloc(test_context.runtime->get_vk_resource());

	Unique_view<BufferLike<float>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
	for (int i = 0; i < 4; i++) {
		foo[i] = (i + 1);
	}

	auto buf0 = vuk::acquire("b0", foo.get(), vuk::Access::eNone);

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 460
#pragma shader_stage(compute)
#include <runtime>

layout (push_constant) uniform data {
	REF(float) data_in;
};

layout (local_size_x = 1) in;

void main() {
	ARRAY(data_in)[gl_GlobalInvocationID.x] *= 2;
}
)")));
	pass(4, 1, 1, buf0);
	buf0.wait(*test_context.allocator, test_context.compiler);
	auto test = { 2.f, 4.f, 6.f, 8.f };
	auto schpen = std::span(&foo[0], 4);
	CHECK(schpen == std::span(test));
}
/*
TEST_CASE("shader buffer access (ptr)") {
  Allocator alloc(test_context.runtime->get_vk_resource());

  Unique_ptr<BufferLike<float>> foo = *allocate_array<float>(alloc, 4, MemoryUsage::eCPUonly);
  for (int i = 0; i < 4; i++) {
    foo[i] = (i + 1);
  }

  auto buf0 = vuk::acquire("b0", foo.get(), vuk::Access::eNone);

  auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 460
#pragma shader_stage(compute)
#include <runtime>

layout (std430, binding = 0) buffer coherent BufferIn {
  float[] data_in;
};

layout (local_size_x = 1) in;

void main() {
  data_in[gl_GlobalInvocationID.x] *= 2;
}
)")));
  pass(4, 1, 1, buf0);
  buf0.wait(*test_context.allocator, test_context.compiler);
  auto test = { 2.f, 4.f, 6.f, 8.f };
  auto schpen = std::span(&foo[0], 4);
  CHECK(schpen == std::span(test));
}*/
