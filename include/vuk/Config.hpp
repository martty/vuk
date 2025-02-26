#pragma once

#ifndef VUK_CUSTOM_VULKAN_HEADER
#include <vulkan/vulkan.h>
#else
#include VUK_CUSTOM_VULKAN_HEADER
#endif

// number of sets that can be bound to the command buffer
#ifndef VUK_MAX_SETS
#define VUK_MAX_SETS 8u
#endif

// number of bindings (individual descriptor) per set for non-persistent descriptorsets
#ifndef VUK_MAX_BINDINGS
#define VUK_MAX_BINDINGS 16u
#endif

// number of attributes that can be bound to the command buffer
#ifndef VUK_MAX_ATTRIBUTES
#define VUK_MAX_ATTRIBUTES 8u
#endif

// number of color attachments supported
#ifndef VUK_MAX_COLOR_ATTACHMENTS
#define VUK_MAX_COLOR_ATTACHMENTS 8u
#endif

// size of the push constant buffer
#ifndef VUK_MAX_PUSHCONSTANT_SIZE
#define VUK_MAX_PUSHCONSTANT_SIZE 128u
#endif

// number of individual push constant ranges that can be bound to the command buffer
#ifndef VUK_MAX_PUSHCONSTANT_RANGES
#define VUK_MAX_PUSHCONSTANT_RANGES 8u
#endif

// number of specialization constants that can be set per pipeline
#ifndef VUK_MAX_SPECIALIZATIONCONSTANT_RANGES
#define VUK_MAX_SPECIALIZATIONCONSTANT_RANGES 64u
#endif

// number of bytes specialization constants can take up for pipelines
#ifndef VUK_MAX_SPECIALIZATIONCONSTANT_SIZE
#define VUK_MAX_SPECIALIZATIONCONSTANT_SIZE 32u
#endif

// number of viewports that can be set on the command buffer
#ifndef VUK_MAX_VIEWPORTS
#define VUK_MAX_VIEWPORTS 1u
#endif

// number of scissors that can be set on the command buffer
#ifndef VUK_MAX_SCISSORS
#define VUK_MAX_SCISSORS 1u
#endif

#ifndef VUK_DISABLE_EXCEPTIONS
#define VUK_USE_EXCEPTIONS 1
#else
#define VUK_USE_EXCEPTIONS 0
#endif

#if VUK_COMPILER_CLANGPP || VUK_COMPILER_CLANGCL || VUK_COMPILER_GPP
#define VUK_UNREACHABLE(msg) (assert(false && msg), __builtin_unreachable())
#elif VUK_COMPILER_MSVC
#define VUK_UNREACHABLE(msg) (assert(false && msg), __assume(0))
#else
#define VUK_UNREACHABLE(msg) assert(false && msg)
#endif
