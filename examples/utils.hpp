#pragma once

#include <fstream>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace util {
	struct Vertex {
		glm::vec3 position;
		glm::vec3 normal;
		glm::vec3 tangent;
		glm::vec3 bitangent;
		glm::vec2 uv_coordinates;
	};

	using Mesh = std::pair<std::vector<Vertex>, std::vector<unsigned>>;

	inline Mesh generate_cube() {
		// clang-format off
		return Mesh(std::vector<Vertex> {
			// back
			Vertex{ {-1, -1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {1, 1} }, Vertex{ {1, 1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, 0} },
				Vertex{ {1, -1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, 1} }, Vertex{ {1, 1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, 0} },
				Vertex{ {-1, -1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {1, 1} }, Vertex{ {-1, 1, -1}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {1, 0} },
				// front 
				Vertex{ {-1, -1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {0, 1} }, Vertex{ {1, -1, 1}, {0, 0, 1},{1, 0.0, 0}, {0, 1, 0}, {1, 1} },
				Vertex{ {1, 1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {1, 0} }, Vertex{ {1, 1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {1, 0} },
				Vertex{ {-1, 1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {0, 0} }, Vertex{ {-1, -1, 1}, {0, 0, 1}, {1, 0.0, 0}, {0, 1, 0}, {0, 1} },
				// left 
				Vertex{ {-1, 1, -1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 0} }, Vertex{ {-1, -1, -1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1} },
				Vertex{ {-1, 1, 1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0} }, Vertex{ {-1, -1, -1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1} },
				Vertex{ {-1, -1, 1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 1} }, Vertex{ {-1, 1, 1}, {-1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0} },
				// right 
				Vertex{ {1, 1, 1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {0, 0} }, Vertex{ {1, -1, -1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {1, 1} },
				Vertex{ {1, 1, -1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {1, 0} }, Vertex{ {1, -1, -1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {1, 1} },
				Vertex{ {1, 1, 1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {0, 0} }, Vertex{ {1, -1, 1}, {1, 0, 0}, {0, 0, -1}, {0, 1, 0}, {0, 1} },
				// bottom 
				Vertex{ {-1, -1, -1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1} }, Vertex{ {1, -1, -1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {1, 1} },
				Vertex{ {1, -1, 1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {1, 0} }, Vertex{ {1, -1, 1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {1, 0} },
				Vertex{ {-1, -1, 1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 0} }, Vertex{ {-1, -1, -1}, {0, -1, 0}, {1, 0, 0}, {0, 0, 1}, {0, 1} },
				// top 
				Vertex{ {-1, 1, -1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0} }, Vertex{ {1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {1, 1} },
				Vertex{ {1, 1, -1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {1, 0} }, Vertex{ {1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {1, 1} },
				Vertex{ {-1, 1, -1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0, 0} }, Vertex{ {-1, 1, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}, {0, 1} } },
			{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 });
		// clang-format on
	}

	inline std::string read_entire_file(const std::string& path) {
		std::ostringstream buf;
		std::ifstream input(path.c_str());
		assert(input);
		buf << input.rdbuf();
		return buf.str();
	}

	inline std::vector<uint32_t> read_spirv(const std::string& path) {
		std::ostringstream buf;
		std::ifstream input(path.c_str(), std::ios::ate | std::ios::binary);
		assert(input);
		size_t file_size = (size_t)input.tellg();
		std::vector<uint32_t> buffer(file_size / sizeof(uint32_t));
		input.seekg(0);
		input.read(reinterpret_cast<char*>(buffer.data()), file_size);
		return buffer;
	}
} // namespace util