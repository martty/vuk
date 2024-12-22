R"(
// INCLUDE THIS FILE BY INCLUDING <runtime>

// cribbed from https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/vulkan/runtime/bvh/vk_build_helpers.h
/*
 * Copyright © 2022 Konstantin Seurer
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require


#define TYPE(type, align)                                                                                              \
   layout(buffer_reference, buffer_reference_align = align, scalar) buffer type##_ref                                  \
   {                                                                                                                   \
      type value;                                                                                                      \
   };

#define REF(type)  type##_ref
#define VOID_REF   uint64_t
#define NULL       0
#define DEREF(var) var.value
#define SIZEOF(type) uint32_t(uint64_t(REF(type)(uint64_t(0)) + 1))

#define OFFSET(ptr, offset) (uint64_t(ptr) + offset)

#define INFINITY (1.0 / 0.0)
#define NAN      (0.0 / 0.0)

#define INDEX(type, ptr, index) REF(type)(OFFSET(ptr, (index)*SIZEOF(type)))

TYPE(int8_t, 1);
TYPE(uint8_t, 1);
TYPE(int16_t, 2);
TYPE(uint16_t, 2);
TYPE(int32_t, 4);
TYPE(uint32_t, 4);
TYPE(int64_t, 8);
TYPE(uint64_t, 8);

TYPE(float, 4);

TYPE(vec2, 4);
TYPE(vec3, 4);
TYPE(vec4, 4);

TYPE(uvec4, 16);

TYPE(VOID_REF, 8);)"