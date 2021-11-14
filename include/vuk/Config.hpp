#pragma once

#ifndef VUK_CUSTOM_VULKAN_HEADER
#include <vulkan/vulkan.h>
#else
#include VUK_CUSTOM_VULKAN_HEADER
#endif

#define VUK_MAX_SETS 8u
#define VUK_MAX_BINDINGS 16u
#define VUK_MAX_ATTRIBUTES 8u
#define VUK_MAX_COLOR_ATTACHMENTS 8u
#define VUK_MAX_PUSHCONSTANT_RANGES 8u
#define VUK_MAX_SPECIALIZATIONCONSTANT_RANGES 8u
// number of bytes all specialization constants can take up
#define VUK_MAX_SPECIALIZATIONCONSTANT_DATA 32u
#define VUK_MAX_VIEWPORTS 1u
#define VUK_MAX_SCISSORS 1u