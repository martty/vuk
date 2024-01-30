#include <shady.h>

location(0) input vec3 inColor;

location(0) output vec4 fragColor;

fragment_shader void main() {
	fragColor = (vec4){ inColor.x, inColor.y, inColor.z, 1.0f };
}