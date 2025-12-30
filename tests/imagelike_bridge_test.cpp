#include "vuk/runtime/vk/Allocation.hpp"
#include <doctest/doctest.h>

using namespace vuk;

TEST_CASE("ImageLike - R32G32B32A32Sfloat format") {
	ImageLike<Format::eR32G32B32A32Sfloat> pixel{ 1.0f, 0.5f, 0.25f, 1.0f };

	SUBCASE("Format properties") {
		CHECK(ImageLike<Format::eR32G32B32A32Sfloat>::size_bytes == 16);
		CHECK(ImageLike<Format::eR32G32B32A32Sfloat>::component_count == 4);
		CHECK(ImageLike<Format::eR32G32B32A32Sfloat>::cdt == ComponentDataType::eFloat32);
	}

	SUBCASE("Component access via methods") {
		CHECK(pixel.r() == 1.0f);
		CHECK(pixel.g() == 0.5f);
		CHECK(pixel.b() == 0.25f);
		CHECK(pixel.a() == 1.0f);
	}

	SUBCASE("Component access via array") {
		CHECK(pixel[0] == 1.0f);
		CHECK(pixel[1] == 0.5f);
		CHECK(pixel[2] == 0.25f);
		CHECK(pixel[3] == 1.0f);
	}

	SUBCASE("Component access via data array") {
		CHECK(pixel.data[0] == 1.0f);
		CHECK(pixel.data[1] == 0.5f);
		CHECK(pixel.data[2] == 0.25f);
		CHECK(pixel.data[3] == 1.0f);
	}

	SUBCASE("Modify components") {
		pixel.r(0.75f);
		CHECK(pixel.r() == 0.75f);
		CHECK(pixel[0] == 0.75f);

		pixel[1] = 0.6f;
		CHECK(pixel.g() == 0.6f);
	}
}

TEST_CASE("ImageLike - R8G8B8A8Unorm normalized format") {
	SUBCASE("Format properties") {
		CHECK(ImageLike<Format::eR8G8B8A8Unorm>::size_bytes == 4);
		CHECK(ImageLike<Format::eR8G8B8A8Unorm>::component_count == 4);
		CHECK(ImageLike<Format::eR8G8B8A8Unorm>::cdt == ComponentDataType::eUnorm8);
	}

	SUBCASE("Construction from raw values") {
		ImageLike<Format::eR8G8B8A8Unorm> pixel{ uint8_t(255), uint8_t(128), uint8_t(0), uint8_t(255) };
		
		// Raw storage
		CHECK(pixel.data[0] == 255);
		CHECK(pixel.data[1] == 128);
		CHECK(pixel.data[2] == 0);
		CHECK(pixel.data[3] == 255);
	}

	SUBCASE("Normalized conversion - reading") {
		ImageLike<Format::eR8G8B8A8Unorm> pixel{ uint8_t(255), uint8_t(128), uint8_t(0), uint8_t(255) };
		
		// Reading converts to [0, 1] float
		CHECK(pixel.r() == doctest::Approx(1.0f));
		CHECK(pixel.g() == doctest::Approx(0.5f).epsilon(0.01f));
		CHECK(pixel.b() == doctest::Approx(0.0f));
		CHECK(pixel.a() == doctest::Approx(1.0f));
	}

	SUBCASE("Normalized conversion - writing") {
		ImageLike<Format::eR8G8B8A8Unorm> pixel{};
		
		// Writing converts from [0, 1] float to uint8
		pixel.r(1.0f);
		pixel.g(0.5f);
		pixel.b(0.0f);
		pixel.a(1.0f);
		
		CHECK(pixel.data[0] == 255);
		CHECK(pixel.data[1] == 128); // 0.5 * 255 + 0.5 = 128
		CHECK(pixel.data[2] == 0);
		CHECK(pixel.data[3] == 255);
	}

	SUBCASE("Packed uint32 construction") {
		// ABGR format: 0xAABBGGRR
		ImageLike<Format::eR8G8B8A8Unorm> red(0xFF0000FFu);    // Opaque red
		ImageLike<Format::eR8G8B8A8Unorm> green(0xFF00FF00u);  // Opaque green
		ImageLike<Format::eR8G8B8A8Unorm> blue(0xFFFF0000u);   // Opaque blue
		ImageLike<Format::eR8G8B8A8Unorm> white(0xFFFFFFFFu);  // Opaque white
		
		CHECK(red.data[0] == 255);   // R
		CHECK(red.data[1] == 0);     // G
		CHECK(red.data[2] == 0);     // B
		CHECK(red.data[3] == 255);   // A
		
		CHECK(green.data[0] == 0);
		CHECK(green.data[1] == 255);
		CHECK(green.data[2] == 0);
		CHECK(green.data[3] == 255);
		
		CHECK(white.data[0] == 255);
		CHECK(white.data[1] == 255);
		CHECK(white.data[2] == 255);
		CHECK(white.data[3] == 255);
	}

	SUBCASE("to_packed() conversion") {
		ImageLike<Format::eR8G8B8A8Unorm> pixel{ uint8_t(255), uint8_t(0), uint8_t(0), uint8_t(255) };
		uint32_t packed = pixel.to_packed();
		CHECK(packed == 0xFF0000FFu); // ABGR format
	}
}

TEST_CASE("ImageLike - R8G8B8A8Srgb gamma correction") {
	SUBCASE("Format properties") {
		CHECK(ImageLike<Format::eR8G8B8A8Srgb>::size_bytes == 4);
		CHECK(ImageLike<Format::eR8G8B8A8Srgb>::component_count == 4);
		CHECK(ImageLike<Format::eR8G8B8A8Srgb>::cdt == ComponentDataType::eSrgb8);
	}

	SUBCASE("sRGB to linear conversion") {
		ImageLike<Format::eR8G8B8A8Srgb> pixel{ uint8_t(188), uint8_t(188), uint8_t(188), uint8_t(255) };
		
		// Reading RGB converts from sRGB to linear (approximately 0.5 linear)
		float r = pixel.r();
		float g = pixel.g();
		float b = pixel.b();
		
		CHECK(r == doctest::Approx(0.5f).epsilon(0.05f));
		CHECK(g == doctest::Approx(0.5f).epsilon(0.05f));
		CHECK(b == doctest::Approx(0.5f).epsilon(0.05f));
		
		// Alpha is always linear
		CHECK(pixel.a() == doctest::Approx(1.0f));
	}

	SUBCASE("Linear to sRGB conversion") {
		ImageLike<Format::eR8G8B8A8Srgb> pixel{};
		
		// Writing RGB converts from linear to sRGB
		pixel.r(0.5f);
		pixel.g(0.5f);
		pixel.b(0.5f);
		pixel.a(1.0f); // Alpha remains linear
		
		// 0.5 linear should be approximately 188 in sRGB
		CHECK(pixel.data[0] == doctest::Approx(188).epsilon(2));
		CHECK(pixel.data[1] == doctest::Approx(188).epsilon(2));
		CHECK(pixel.data[2] == doctest::Approx(188).epsilon(2));
		CHECK(pixel.data[3] == 255); // Alpha is linear
	}

	SUBCASE("Alpha channel is always linear in sRGB") {
		ImageLike<Format::eR8G8B8A8Srgb> pixel{};
		
		pixel.a(0.5f);
		CHECK(pixel.data[3] == 128); // 0.5 * 255 + 0.5, no gamma correction
		
		pixel.data[3] = 128;
		CHECK(pixel.a() == doctest::Approx(0.5f).epsilon(0.01f)); // Linear read
	}
}

TEST_CASE("ImageLike - R16G16Unorm normalized format") {
	SUBCASE("Format properties") {
		CHECK(ImageLike<Format::eR16G16Unorm>::size_bytes == 4);
		CHECK(ImageLike<Format::eR16G16Unorm>::component_count == 2);
		CHECK(ImageLike<Format::eR16G16Unorm>::cdt == ComponentDataType::eUnorm16);
	}

	SUBCASE("16-bit normalized conversion") {
		ImageLike<Format::eR16G16Unorm> pixel{ uint16_t(65535), uint16_t(32768) };
		
		CHECK(pixel.r() == doctest::Approx(1.0f));
		CHECK(pixel.g() == doctest::Approx(0.5f).epsilon(0.001f));
	}

	SUBCASE("Writing normalized values") {
		ImageLike<Format::eR16G16Unorm> pixel{};
		pixel.r(1.0f);
		pixel.g(0.5f);
		
		CHECK(pixel.data[0] == 65535);
		CHECK(pixel.data[1] == doctest::Approx(32768).epsilon(1));
	}
}

TEST_CASE("ImageLike - R8Snorm signed normalized format") {
	SUBCASE("Format properties") {
		CHECK(ImageLike<Format::eR8Snorm>::size_bytes == 1);
		CHECK(ImageLike<Format::eR8Snorm>::component_count == 1);
		CHECK(ImageLike<Format::eR8Snorm>::cdt == ComponentDataType::eSnorm8);
	}

	SUBCASE("Signed normalized conversion") {
		ImageLike<Format::eR8Snorm> pixel{ int8_t(127) };
		CHECK(pixel.r() == doctest::Approx(1.0f));
		
		pixel.data[0] = -127;
		CHECK(pixel.r() == doctest::Approx(-1.0f));
		
		pixel.data[0] = 0;
		CHECK(pixel.r() == doctest::Approx(0.0f));
	}

	SUBCASE("Writing signed normalized values") {
		ImageLike<Format::eR8Snorm> pixel{};
		pixel.r(1.0f);
		CHECK(pixel.data[0] == 127);
		
		pixel.r(-1.0f);
		CHECK(pixel.data[0] == -127);
		
		pixel.r(0.0f);
		CHECK(pixel.data[0] == 0);
	}
}

TEST_CASE("ImageLike - R32Uint single component") {
	ImageLike<Format::eR32Uint> pixel{ 42u };

	SUBCASE("Format properties") {
		CHECK(ImageLike<Format::eR32Uint>::size_bytes == 4);
		CHECK(ImageLike<Format::eR32Uint>::component_count == 1);
	}

	SUBCASE("Component access") {
		CHECK(pixel.r() == 42u);
		CHECK(pixel[0] == 42u);
		CHECK(pixel.data[0] == 42u);
	}
}

TEST_CASE("ImageLike - R32G32Sfloat two components") {
	ImageLike<Format::eR32G32Sfloat> pixel{ 1.5f, 2.5f };

	SUBCASE("Format properties") {
		CHECK(ImageLike<Format::eR32G32Sfloat>::size_bytes == 8);
		CHECK(ImageLike<Format::eR32G32Sfloat>::component_count == 2);
	}

	SUBCASE("Component access") {
		CHECK(pixel.r() == 1.5f);
		CHECK(pixel.g() == 2.5f);
		CHECK(pixel[0] == 1.5f);
		CHECK(pixel[1] == 2.5f);
	}

	SUBCASE("Component modification") {
		pixel.r(3.5f);
		CHECK(pixel[0] == 3.5f);

		pixel.g(4.5f);
		CHECK(pixel[1] == 4.5f);
	}
}

TEST_CASE("ImageLike - R32G32B32Uint three components") {
	ImageLike<Format::eR32G32B32Uint> pixel{ 10u, 20u, 30u };

	SUBCASE("Format properties") {
		CHECK(ImageLike<Format::eR32G32B32Uint>::size_bytes == 12);
		CHECK(ImageLike<Format::eR32G32B32Uint>::component_count == 3);
	}

	SUBCASE("Component access") {
		CHECK(pixel.r() == 10u);
		CHECK(pixel.g() == 20u);
		CHECK(pixel.b() == 30u);
		CHECK(pixel[0] == 10u);
		CHECK(pixel[1] == 20u);
		CHECK(pixel[2] == 30u);
	}
}

TEST_CASE("ImageLike - color mixing operations") {
	ImageLike<Format::eR8G8B8A8Unorm> red(0xFF0000FFu);   // Opaque red
	ImageLike<Format::eR8G8B8A8Unorm> green(0xFF00FF00u); // Opaque green

	// Mix colors using normalized float values
	ImageLike<Format::eR8G8B8A8Unorm> mixed{};
	mixed.r((red.r() + green.r()) * 0.5f);
	mixed.g((red.g() + green.g()) * 0.5f);
	mixed.b((red.b() + green.b()) * 0.5f);
	mixed.a(1.0f);

	CHECK(mixed.r() == doctest::Approx(0.5f).epsilon(0.01f));
	CHECK(mixed.g() == doctest::Approx(0.5f).epsilon(0.01f));
	CHECK(mixed.b() == doctest::Approx(0.0f));
	CHECK(mixed.a() == doctest::Approx(1.0f));
}

TEST_CASE("ImageLike - sRGB vs UNORM comparison") {
	// Compare gamma correction behavior
	ImageLike<Format::eR8G8B8A8Unorm> unorm{ uint8_t(128), uint8_t(128), uint8_t(128), uint8_t(255) };
	ImageLike<Format::eR8G8B8A8Srgb> srgb{ uint8_t(128), uint8_t(128), uint8_t(128), uint8_t(255) };

	// UNORM: linear mapping
	float unorm_value = unorm.r();
	CHECK(unorm_value == doctest::Approx(0.5f).epsilon(0.01f));

	// sRGB: gamma correction applied (128/255 sRGB is darker than 0.5 linear)
	float srgb_value = srgb.r();
	CHECK(srgb_value < unorm_value); // sRGB should be darker
	CHECK(srgb_value == doctest::Approx(0.215f).epsilon(0.05f)); // Approximately 0.215 linear
}
