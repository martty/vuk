#pragma once

namespace vuk {
	struct Stream {
		Stream(Allocator alloc, Executor* executor) : alloc(alloc), executor(executor) {}
		virtual ~Stream() {}
		Allocator alloc;
		Executor* executor = nullptr;
		DomainFlagBits domain;
		std::vector<Stream*> dependencies;
		std::vector<Signal*> dependent_signals;

		virtual void add_dependency(Stream* dep) = 0;
		virtual void sync_deps() = 0;

		virtual Signal* make_signal() = 0;

		void add_dependent_signal(Signal* signal) {
			dependent_signals.push_back(signal);
		}

		virtual void synch_image(ImageAttachment& img_att, Subrange::Image subrange, StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) = 0;
		virtual void synch_memory(StreamResourceUse src_use, StreamResourceUse dst_use, void* tag) = 0;

		struct SubmitResult {
			VkSemaphore sema_wait;
		};

		virtual Result<SubmitResult> submit() = 0;
	};
}