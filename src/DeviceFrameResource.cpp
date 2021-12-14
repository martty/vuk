#include "vuk/Context.hpp"
#include "vuk/resources/DeviceFrameResource.hpp"

namespace vuk {
	DeviceSuperFrameResource::DeviceSuperFrameResource(Context& ctx, uint64_t frames_in_flight) : direct(ctx, ctx.get_legacy_gpu_allocator()), frames_in_flight(frames_in_flight) {
		frames_storage = std::unique_ptr<char[]>(new char[sizeof(DeviceFrameResource) * frames_in_flight]);
		for (uint64_t i = 0; i < frames_in_flight; i++) {
			new(frames_storage.get() + i * sizeof(DeviceFrameResource)) DeviceFrameResource(direct.device, *this);
		}
		frames = reinterpret_cast<DeviceFrameResource*>(frames_storage.get());
	}

	DeviceFrameResource& DeviceSuperFrameResource::get_next_frame() {
		std::unique_lock _(new_frame_mutex);
		auto& ctx = direct.get_context();
		frame_counter++;
		local_frame = frame_counter % frames_in_flight;

		auto& f = frames[local_frame];
		f.wait();
		deallocate_frame(f);
		f.current_frame = frame_counter.load();

		return f;
	}

	void DeviceSuperFrameResource::deallocate_frame(DeviceFrameResource& f) {
		//f.descriptor_set_cache.collect(frame_counter.load(), 16);
		direct.deallocate_semaphores(f.semaphores);
		direct.deallocate_fences(f.fences);
		for (auto& c : f.cmdbuffers_to_free) {
			direct.deallocate_commandbuffers(c.command_pool, std::span{ &c.command_buffer, 1 });
		}
		direct.deallocate_commandpools(f.cmdpools_to_free);
		direct.deallocate_buffers(f.buffer_gpus);
		direct.deallocate_buffers(f.buffer_cross_devices);
		direct.deallocate_framebuffers(f.framebuffers);
		direct.deallocate_images(f.images);
		direct.deallocate_image_views(f.image_views);
		direct.deallocate_persistent_descriptor_sets(f.persistent_descriptor_sets);
		direct.deallocate_descriptor_sets(f.descriptor_sets);
		direct.ctx->make_timestamp_results_available(f.ts_query_pools);
		direct.deallocate_timestamp_query_pools(f.ts_query_pools);

		f.semaphores.clear();
		f.fences.clear();
		f.buffer_cross_devices.clear();
		f.buffer_gpus.clear();
		f.cmdbuffers_to_free.clear();
		f.cmdpools_to_free.clear();
		auto& legacy = direct.legacy_gpu_allocator;
		legacy->reset_pool(f.linear_cpu_only);
		legacy->reset_pool(f.linear_cpu_gpu);
		legacy->reset_pool(f.linear_gpu_cpu);
		legacy->reset_pool(f.linear_gpu_only);
		f.framebuffers.clear();
		f.images.clear();
		f.image_views.clear();
		f.persistent_descriptor_sets.clear();
		f.descriptor_sets.clear();
		f.ts_query_pools.clear();
		f.query_index = 0;
	}
}