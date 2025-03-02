#pragma once

#include <cstdint>

// Support header for embedding data

struct EmbeddedRes {
	const char* data;
	std::size_t size;
};

#define VUK_EMBEDDED_RESOURCE(name) extern "C" EmbeddedRes name()

// declare the resource first:
// VUK_EMBEDDED_RESOURCE(imgui_vert);
// then:
// imgui_vert() returns an EmbeddedRes with the resource data and size