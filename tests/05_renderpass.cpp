#include "TestContext.hpp"
#include "vuk/runtime/vk/AllocatorHelpers.hpp"
#include "vuk/vsl/Core.hpp"
#include <doctest/doctest.h>

using namespace vuk;

namespace {
	auto image2buf = make_pass("copy image to buffer", [](CommandBuffer& cbuf, VUK_IA(Access::eTransferRead) src, VUK_BA(Access::eTransferWrite) dst) {
		BufferImageCopy bc;
		bc.imageOffset = { 0, 0, 0 };
		bc.bufferRowLength = 0;
		bc.bufferImageHeight = 0;
		bc.imageExtent = src->extent;
		bc.imageSubresource.aspectMask = format_to_aspect(src->format);
		bc.imageSubresource.mipLevel = src->base_level;
		bc.imageSubresource.baseArrayLayer = src->base_layer;
		assert(src->layer_count == 1); // unsupported yet
		bc.imageSubresource.layerCount = src->layer_count;
		bc.bufferOffset = dst->offset;
		cbuf.copy_image_to_buffer(src, dst, bc);
		return dst;
	});
} // namespace

TEST_CASE("renderpass clear") {
	{
		auto rpclear = make_pass("rp clear", [](CommandBuffer& cbuf, VUK_IA(Access::eColorWrite) dst) {
			cbuf.clear_image(dst, ClearColor(5u, 5u, 5u, 5u));
			return dst;
		});
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		ia.level_count = 1;
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
		auto fut2 = rpclear(fut);
		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(image2buf(fut2, dst_buf)).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
	}
}

TEST_CASE("renderpass framebuffer inference") {
	{
		auto rpclear = make_pass("rp clear", [](CommandBuffer& cbuf, VUK_IA(Access::eColorWrite) dst, VUK_IA(Access::eDepthStencilRW)) {
			cbuf.clear_image(dst, ClearColor(5u, 5u, 5u, 5u));
			return dst;
		});
		auto data = { 1u, 2u, 3u, 4u };
		auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
		ia.level_count = 1;
		auto [img, fut] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

		size_t alignment = format_to_texel_block_size(fut->format);
		size_t size = compute_image_size(fut->format, fut->extent);
		auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });

		auto depth_img = declare_ia("depth");
		depth_img->format = vuk::Format::eD32Sfloat;

		auto fut2 = rpclear(fut, std::move(depth_img));
		auto dst_buf = discard_buf("dst", *dst);
		auto res = download_buffer(image2buf(fut2, dst_buf)).get(*test_context.allocator, test_context.compiler);
		auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
		CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 5; }));
	}
}

TEST_CASE("buffer size inference") {
	auto data = { 1u, 2u, 3u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));
	auto buf1 = declare_buf("b1");
	buf1->memory_usage = MemoryUsage::eGPUonly;
	buf1.same_size(buf0);
	auto buf2 = declare_buf("b2");
	buf2->memory_usage = MemoryUsage::eGPUonly;
	buf2.same_size(buf1);
	auto buf3 = declare_buf("b3");
	buf3.same_size(buf2);
	buf3->memory_usage = MemoryUsage::eGPUonly;

	auto res = download_buffer(copy(std::move(buf0), std::move(buf3))).get(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
}

TEST_CASE("buffer size with inference with math") {
	auto data = { 1u, 2u, 3u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));
	auto buf1 = declare_buf("b1");
	buf1->memory_usage = MemoryUsage::eGPUonly;
	buf1.same_size(buf0);
	auto buf2 = declare_buf("b2");
	buf2->memory_usage = MemoryUsage::eGPUonly;
	buf2.same_size(buf1);
	auto buf3 = declare_buf("b3");
	buf3.set_size(buf2.get_size() * 2);
	buf3->memory_usage = MemoryUsage::eGPUonly;

	auto data2 = { 1u, 2u, 3u, 4u, 5u, 6u };
	auto [b4, buf4] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data2));

	auto res = download_buffer(copy(std::move(buf4), std::move(buf3))).get(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
}

TEST_CASE("extract-convergence use") {
	auto data = { 1u, 2u, 3u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));
	auto buf1 = declare_buf("b1");
	buf1->memory_usage = MemoryUsage::eGPUonly;
	buf1.same_size(buf0);
	auto buf2 = declare_buf("b2");
	buf2->memory_usage = MemoryUsage::eGPUonly;
	auto size = buf2.get_size();
	buf2.same_size(buf1);
	auto buf3 = declare_buf("b3");
	copy(buf2, buf1);
	buf3.set_size(size * 2);
	buf3->memory_usage = MemoryUsage::eGPUonly;

	auto data2 = { 1u, 2u, 3u, 4u, 5u, 6u };
	auto [b4, buf4] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data2));

	auto res = download_buffer(copy(std::move(buf4), std::move(buf3))).get(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
}

auto use_all = make_pass("use 3", [](CommandBuffer& cbuf, VUK_BA(Access::eTransferWrite), VUK_BA(Access::eTransferWrite), VUK_BA(Access::eTransferWrite)) {});

TEST_CASE("extract-extract") {
	auto data = { 1u, 2u, 3u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));
	auto buf1 = declare_buf("b1");
	buf1->memory_usage = MemoryUsage::eGPUonly;
	auto sz = buf0->sz_bytes;
	buf1.set_size(sz);
	auto buf2 = declare_buf("b2");
	buf2->memory_usage = MemoryUsage::eGPUonly;
	buf2.set_size(buf0.get_size());

	auto data2 = { 1u, 2u, 3u };
	auto [b4, buf4] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data2));

	use_all(buf0, buf1, buf2);
	auto res = download_buffer(copy(std::move(buf4), std::move(buf2))).get(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));
}

TEST_CASE("lift compute") {
	auto data = { 1u, 2u, 3u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));
	vuk::PipelineBaseCreateInfo pbci;
	pbci.add_glsl(R"(#version 450
#pragma shader_stage(compute)

layout (std430, binding = 0) buffer coherent BufferIn {
	uint[] data_in;
};

layout (local_size_x = 1) in;

void main() {
	data_in[gl_GlobalInvocationID.x] *= 2;
}
)",
	              "<>");
	auto pass = lift_compute(test_context.runtime->get_pipeline(pbci));
	pass(3, 1, 1, buf0);
	auto res = download_buffer(buf0).get(*test_context.allocator, test_context.compiler);
	auto test = { 2u, 4u, 6u };
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(test));
}

TEST_CASE("lift compute 2") {
	auto data = { 1u, 2u, 3u, 4u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));

	auto data2 = { 4u, 4u, 2u, 2u };
	auto [b1, buf1] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data2));
	vuk::PipelineBaseCreateInfo pbci;
	pbci.add_glsl(R"(#version 450
#pragma shader_stage(compute)

layout (std430, binding = 0) buffer coherent BufferIn {
	uint[] data_in;
};

layout (binding = 1) uniform BufferIn2 {
	uvec4 data_in2;
};

layout (local_size_x = 1) in;

void main() {
	data_in[gl_GlobalInvocationID.x] *= data_in2[gl_GlobalInvocationID.x];
}
)",
	              "<>");
	auto pass = lift_compute(test_context.runtime->get_pipeline(pbci));
	pass(4, 1, 1, buf0, buf1);
	auto res = download_buffer(buf0).get(*test_context.allocator, test_context.compiler);
	auto test = { 4u, 8u, 6u, 8u };
	CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(test));
}

TEST_CASE("lift compute 3") {
	auto data = { 1u, 2u, 3u, 4u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));

	auto data2 = { 4u, 4u, 2u, 2u };
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
	auto [img, img0] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data2));

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 450
#pragma shader_stage(compute)

layout (std430, binding = 0) buffer coherent BufferIn {
	uint[] data_in;
};

uniform layout(binding=1,r32ui) readonly uimage2D someImage;

layout (local_size_x = 1) in;

void main() {
	data_in[gl_GlobalInvocationID.x] *= imageLoad(someImage, ivec2(gl_GlobalInvocationID.x % 2,gl_GlobalInvocationID.x / 2)).x;
}
)")));
	pass(4, 1, 1, buf0, img0);
	auto res = download_buffer(buf0).get(*test_context.allocator, test_context.compiler);
	auto test = { 4u, 8u, 6u, 8u };
	CHECK(std::span((uint32_t*)res->mapped_ptr, 4) == std::span(test));
}

TEST_CASE("separate sampler") {
	auto data2 = { 4.f, 4.f, 2.f, 2.f };
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 2, 2, 1 }, Samples::e1);
	auto [img, img0] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data2));

	auto nearest_samp = vuk::acquire_sampler("nearest", SamplerCreateInfo{ .magFilter = vuk::Filter::eNearest, .minFilter = vuk::Filter::eNearest });
	auto linear_samp = vuk::acquire_sampler("linear", SamplerCreateInfo{ .magFilter = vuk::Filter::eLinear, .minFilter = vuk::Filter::eLinear });

	auto out_nearest =
	    vuk::declare_ia("out_nearest", ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 2, 2, 1 }, Samples::e1));
	auto out_linear =
	    vuk::declare_ia("out_linear", ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 2, 2, 1 }, Samples::e1));

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 450
#pragma shader_stage(compute)

uniform layout(binding=0) texture2D someImage;
uniform layout(binding=1) sampler nearest;
uniform layout(binding=2) sampler linear;

uniform layout(binding=3,r32f) image2D out_nearest;
uniform layout(binding=4,r32f) image2D out_linear;

layout (local_size_x = 1) in;

void main() {
	ivec2 coord = ivec2(gl_GlobalInvocationID.x % 2,gl_GlobalInvocationID.x / 2);
	vec2 normcoord = coord / 2;
	imageStore(out_nearest, coord, texture(sampler2D(someImage, nearest), normcoord));
	imageStore(out_linear, coord, texture(sampler2D(someImage, linear), normcoord));
}
)")));
	pass(4, 1, 1, img0, nearest_samp, linear_samp, out_nearest, out_linear);
	size_t alignment = format_to_texel_block_size(out_nearest->format);
	size_t size = compute_image_size(out_nearest->format, out_nearest->extent);
	auto dst0 = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto dst_buf0 = discard_buf("dst", *dst0);
	auto dst1 = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto dst_buf1 = discard_buf("dst", *dst1);
	auto dl = download_buffer(copy(out_nearest, dst_buf0));
	auto res1 = download_buffer(copy(out_linear, dst_buf1)).get(*test_context.allocator, test_context.compiler);
	auto res0 = dl.get(*test_context.allocator, test_context.compiler);
	auto test0 = { 4.f, 4.f, 4.f, 4.f };
	CHECK(std::span((float*)res0->mapped_ptr, 4) == std::span(test0));
	auto test1 = { 3.f, 3.f, 3.f, 3.f };
	CHECK(std::span((float*)res1->mapped_ptr, 4) == std::span(test1));
}

TEST_CASE("combined sampler") {
	auto data2 = { 4.f, 4.f, 2.f, 2.f };
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 2, 2, 1 }, Samples::e1);
	auto [img, img0] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data2));

	auto nearest_samp = vuk::acquire_sampler("nearest", SamplerCreateInfo{ .magFilter = vuk::Filter::eNearest, .minFilter = vuk::Filter::eNearest });

	auto image_and_samp = vuk::combine_image_sampler("combined", img0, nearest_samp);

	auto out_nearest =
	    vuk::declare_ia("out_nearest", ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Sfloat, { 2, 2, 1 }, Samples::e1));

	auto pass = lift_compute(test_context.runtime->get_pipeline(vuk::PipelineBaseCreateInfo::from_inline_glsl(R"(#version 450
#pragma shader_stage(compute)

uniform layout(binding=0) sampler2D nearest;

uniform layout(binding=3,r32f) image2D out_nearest;

layout (local_size_x = 1) in;

void main() {
	ivec2 coord = ivec2(gl_GlobalInvocationID.x % 2,gl_GlobalInvocationID.x / 2);
	vec2 normcoord = coord / 2;
	imageStore(out_nearest, coord, texture(nearest, normcoord));
}
)")));
	pass(4, 1, 1, image_and_samp, out_nearest);
	size_t alignment = format_to_texel_block_size(out_nearest->format);
	size_t size = compute_image_size(out_nearest->format, out_nearest->extent);
	auto dst0 = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto dst_buf0 = discard_buf("dst", *dst0);
	auto res0 = download_buffer(copy(out_nearest, dst_buf0)).get(*test_context.allocator, test_context.compiler);
	auto test0 = { 4.f, 4.f, 4.f, 4.f };
	CHECK(std::span((float*)res0->mapped_ptr, 4) == std::span(test0));
}

TEST_CASE("lift compute with IR pipeline compile") {
	auto data = { 1u, 2u, 3u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));
	vuk::PipelineBaseCreateInfo pbci;
	pbci.add_glsl(R"(#version 450
#pragma shader_stage(compute)

layout (std430, binding = 0) buffer coherent BufferIn {
	uint[] data_in;
};

layout (local_size_x = 1) in;

void main() {
	data_in[gl_GlobalInvocationID.x] *= 2;
}
)",
	              "<>");
	auto pass = lift_compute(pbci);
	pass(3, 1, 1, buf0);
	auto res = download_buffer(buf0).get(*test_context.allocator, test_context.compiler);
	auto test = { 2u, 4u, 6u };
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(test));
}

TEST_CASE("IR pipeline compile") {
	auto data = { 1u, 2u, 3u };
	auto [b0, buf0] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, DomainFlagBits::eAny, std::span(data));
	vuk::PipelineBaseCreateInfo pbci;
	pbci.add_glsl(R"(#version 450
#pragma shader_stage(compute)

layout (std430, binding = 0) buffer coherent BufferIn {
	uint[] data_in;
};

layout (local_size_x = 1) in;

void main() {
	data_in[gl_GlobalInvocationID.x] *= 2;
}
)",
	              "<>");
	auto pass = make_pass("IR shader", [](CommandBuffer& cbuf, VUK_ARG(PipelineBaseInfo*, Access::eNone) pipe, VUK_BA(Access::eComputeRW) buf) {
		cbuf.bind_compute_pipeline(pipe);
		cbuf.bind_buffer(0, 0, buf);
		cbuf.dispatch(3, 1, 1);
	});
	auto shader = compile_pipeline(pbci);
	pass(shader, buf0);
	auto res = download_buffer(buf0).get(*test_context.allocator, test_context.compiler);
	auto test = { 2u, 4u, 6u };
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(test));
}

TEST_CASE("attachmentless fb") {
	auto data = { 1u, 1u, 1u, 1u };
	auto ia = ImageAttachment::from_preset(ImageAttachment::Preset::eGeneric2D, Format::eR32Uint, { 2, 2, 1 }, Samples::e1);
	ia.level_count = 1;
	auto [img, img_val] = create_image_with_data(*test_context.allocator, DomainFlagBits::eAny, ia, std::span(data));

	auto cleared_img = clear_image(img_val, ClearColor(0u, 0u, 0u, 0u));

	vuk::PipelineBaseCreateInfo pbci;
	pbci.add_glsl(
	    R"(#version 450
			#pragma shader_stage(vertex)
			
			vec2 positions[3] = vec2[](
				vec2(-1.0, -1.0),
				vec2( 3.0, -1.0),
				vec2(-1.0,  3.0)
			);
			
			void main() {
				gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
			}
			)",
	    "<vert>");
	pbci.add_glsl(
	    R"(#version 450
			#pragma shader_stage(fragment)
			
			layout(binding = 0, r32ui) uniform writeonly uimage2D outputImage;
			
			void main() {
				ivec2 coord = ivec2(gl_FragCoord.xy);
				imageStore(outputImage, coord, uvec4(42, 0, 0, 0));
			}
			)",
	    "<frag>");

	// Dispatch a fullscreen triangle in an attachmentless renderpass that writes to a storage image
	auto render_pass = make_pass("attachmentless pass", [pbci](CommandBuffer& cbuf, VUK_IA(Access::eFragmentWrite) output_img) {
		cbuf.set_attachmentless_framebuffer(Extent2D{ 2, 2 }, SampleCountFlagBits::e1);

		// Bind a graphics pipeline that writes to a storage image
		cbuf.bind_graphics_pipeline(test_context.runtime->get_pipeline(pbci));
		cbuf.bind_image(0, 0, output_img);
		cbuf.set_rasterization({});
		cbuf.broadcast_color_blend({});
		cbuf.set_viewport(0, Rect2D::framebuffer());
		cbuf.set_scissor(0, Rect2D::framebuffer());
		cbuf.draw(3, 1, 0, 0);

		return output_img;
	});

	auto rendered_img = render_pass(cleared_img);

	size_t alignment = format_to_texel_block_size(rendered_img->format);
	size_t size = compute_image_size(rendered_img->format, rendered_img->extent);
	auto dst = *allocate_buffer(*test_context.allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, alignment });
	auto dst_buf = discard_buf("dst", *dst);

	auto res = download_buffer(image2buf(rendered_img, dst_buf)).get(*test_context.allocator, test_context.compiler);
	auto updata = std::span((uint32_t*)res->mapped_ptr, 4);
	CHECK(std::all_of(updata.begin(), updata.end(), [](auto& elem) { return elem == 42; }));
}
