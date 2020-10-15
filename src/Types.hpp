#pragma once

#include <vulkan/vulkan.hpp>
#include <Hash.hpp>

namespace vuk {
	struct HandleBase {
		size_t id = UINT64_MAX;
	};

	template<class T>
	struct Handle : public HandleBase {
		T payload;

		bool operator==(const Handle& o) const noexcept {
			return id == o.id;
		}
	};

    class Context;
    template <typename Type>
    class Unique {
        Context* context;
        Type payload;
    public:
        using element_type = Type;

        Unique() : context(nullptr) {}

        explicit Unique(vuk::Context& ctx, Type payload) : context(&ctx), payload(payload) {}
        Unique(Unique const&) = delete;

        Unique(Unique&& other) noexcept : context(other.context), payload(other.release()) {}

        ~Unique() noexcept;

        Unique& operator=(Unique const&) = delete;

        Unique& operator=(Unique&& other) noexcept {
            auto tmp = other.context;
            reset(other.release());
            context = tmp;
            return *this;
        }

        explicit operator bool() const noexcept {
            return payload.operator bool();
        }

        Type const* operator->() const noexcept {
            return &payload;
        }

        Type* operator->() noexcept {
            return &payload;
        }

        Type const& operator*() const noexcept {
            return payload;
        }

        Type& operator*() noexcept {
            return payload;
        }

        const Type& get() const noexcept {
            return payload;
        }

        Type& get() noexcept {
            return payload;
        }

        void reset(Type const& value = Type()) noexcept;

        Type release() noexcept {
            Type value = payload;
            context = nullptr;
            return value;
        }

        void swap(Unique<Type>& rhs) noexcept {
            std::swap(payload, rhs.payload);
            std::swap(context, rhs.context);
        }
    };

    template <typename Type>
    inline void swap(Unique<Type>& lhs, Unique<Type>& rhs) noexcept {
        lhs.swap(rhs);
    }
}

namespace std {
	template<class T>
	struct hash<vuk::Handle<T>> {
		size_t operator()(vuk::Handle<T> const & x) const noexcept {
			size_t h = 0;
			hash_combine(h, x.id, T::objectType);
			return h;
		}
	};

}

namespace vuk {
	using ImageView = Handle<vk::ImageView>;
	using Sampler = Handle<vk::Sampler>;

	struct Buffer {
		vk::DeviceMemory device_memory;
		vk::Buffer buffer;
		size_t offset;
		size_t size;
		void* mapped_ptr;

        bool operator==(const Buffer& o) const {
            return std::tie(device_memory, buffer, offset, size) ==
                std::tie(o.device_memory, o.buffer, o.offset, o.size);
        }
	};

	struct Samples {
		vk::SampleCountFlagBits count;
		bool infer;

		struct Framebuffer {};

		Samples() : count(vk::SampleCountFlagBits::e1), infer(false) {}
		Samples(vk::SampleCountFlagBits samples) : count(samples), infer(false) {}
		Samples(Framebuffer) : infer(true) {}

		constexpr static auto e1 = vk::SampleCountFlagBits::e1;
		constexpr static auto e2 = vk::SampleCountFlagBits::e2;
		constexpr static auto e4 = vk::SampleCountFlagBits::e4;
		constexpr static auto e8 = vk::SampleCountFlagBits::e8;
		constexpr static auto e16 = vk::SampleCountFlagBits::e16;
		constexpr static auto e32 = vk::SampleCountFlagBits::e32;
		constexpr static auto e64 = vk::SampleCountFlagBits::e64;
	};

    struct Texture {
        Unique<vk::Image> image;
        Unique<vuk::ImageView> view;
        vk::Extent3D extent;
        vk::Format format;
        vuk::Samples sample_count;
    };
}
