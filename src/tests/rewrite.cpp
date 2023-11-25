#include "TestContext.hpp"
#include "vuk/AllocatorHelpers.hpp"
#include <doctest/doctest.h>

using namespace vuk;

constexpr bool operator==(const std::span<uint32_t>& lhs, const std::span<uint32_t>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<uint32_t>& lhs, const std::span<const uint32_t>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

constexpr bool operator==(const std::span<const uint32_t>& lhs, const std::span<const uint32_t>& rhs) {
	return std::equal(begin(lhs), end(lhs), begin(rhs), end(rhs));
}

inline TypedFuture<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, const void* src_data, size_t size) {
	// host-mapped buffers just get memcpys
	if (dst.mapped_ptr) {
		memcpy(dst.mapped_ptr, src_data, size);
		return { vuk::declare_buf("_dst", dst) };
	}

	auto src = *allocate_buffer(allocator, BufferCreateInfo{ MemoryUsage::eCPUonly, size, 1 });
	::memcpy(src->mapped_ptr, src_data, size);

	auto src_buf = vuk::declare_buf("_src", *src);
	auto dst_buf = src_buf.rg->make_declare_buffer(dst);
	auto read_ty = src_buf.rg->make_imbued_ty(src_buf.rg->builtin_buffer, Access::eTransferRead);
	auto write_ty = src_buf.rg->make_imbued_ty(src_buf.rg->builtin_buffer, Access::eTransferWrite);
	auto ret_tys = src_buf.rg->make_bound_ty(src_buf.rg->builtin_buffer, 1);
	auto args_ts = { read_ty, write_ty };
	auto call_t = src_buf.rg->make_opaque_fn_ty({ args_ts }, { { ret_tys } }, copy_domain, [size](vuk::CommandBuffer& command_buffer, std::span<void*> args) {
		command_buffer.copy_buffer(*reinterpret_cast<Buffer*>(args[0]), *reinterpret_cast<Buffer*>(args[1]), size);
	});

	auto call_res = src_buf.rg->make_call(call_t, src_buf.head, dst_buf);
	auto result = src_buf.rg->make_release(first(call_res));
	// make_pass("BUFFER UPLOAD", );
	return { src_buf.rg, result };
}

/// @brief Fill a buffer with host data
/// @param allocator Allocator to use for temporary allocations
/// @param copy_domain The domain where the copy should happen (when dst is mapped, the copy happens on host)
/// @param dst Buffer to fill
/// @param data source data
template<class T>
TypedFuture<Buffer> host_data_to_buffer(Allocator& allocator, DomainFlagBits copy_domain, Buffer dst, std::span<T> data) {
	return host_data_to_buffer(allocator, copy_domain, dst, data.data(), data.size_bytes());
}

/// @brief Allocates & fills a buffer with explicitly managed lifetime
/// @param allocator Allocator to allocate this Buffer from
/// @param mem_usage Where to allocate the buffer (host visible buffers will be automatically mapped)
template<class T>
std::pair<Unique<Buffer>, TypedFuture<Buffer>>
create_buffer(Allocator& allocator, vuk::MemoryUsage memory_usage, DomainFlagBits domain, std::span<T> data, size_t alignment = 1) {
	Unique<Buffer> buf(allocator);
	BufferCreateInfo bci{ memory_usage, sizeof(T) * data.size(), alignment };
	auto ret = allocator.allocate_buffers(std::span{ &*buf, 1 }, std::span{ &bci, 1 }); // TODO: dropping error
	Buffer b = buf.get();
	return { std::move(buf), host_data_to_buffer(allocator, domain, b, data) };
}

TEST_CASE("test buffer harness") {
	REQUIRE(test_context.prepare());
	auto data = { 1u, 2u, 3u };
	auto [buf, fut] = create_buffer(*test_context.allocator, MemoryUsage::eGPUonly, vuk::DomainFlagBits::eTransferOnTransfer, std::span(data));
	auto exec = test_context.compiler.link(std::span{ &fut.rg, 1 }, {});
	exec->execute(*test_context.allocator, {});
	/* auto res = fut.get<Buffer>(*test_context.allocator, test_context.compiler);
	CHECK(std::span((uint32_t*)res->mapped_ptr, 3) == std::span(data));*/
}