Extra Features
==============

Vuk provides optional "extra" features that extend the core library with commonly needed functionality. These features are modular and can be enabled selectively via CMake options.

Overview
--------

The extra features are controlled by the ``VUK_EXTRA`` option. When enabled, individual features can be toggled:

- **ImGui Integration** - Render Dear ImGui user interfaces with vuk
- **Simple Init** - Simplified initialization helpers for quick setup
- **SPD (Single Pass Downsampler)** - Efficient mipmap generation

All extra features live in the ``vuk::extra`` namespace and are header-only or optionally compiled.

CMake Configuration
-------------------

Enable extra features in your CMakeLists.txt:

.. code-block:: cmake

   # Enable all extra features (default: ON)
   set(VUK_EXTRA ON)
   
   # Enable ImGui integration (default: ON when VUK_EXTRA is ON)
   set(VUK_EXTRA_IMGUI ON)
   
   # Configure ImGui platform backend (options: glfw, sdl2, sdl3, custom)
   set(VUK_EXTRA_IMGUI_PLATFORM_BACKEND "glfw")
   
   # Enable simple initialization helpers (default: ON when VUK_EXTRA is ON)
   set(VUK_EXTRA_INIT ON)

ImGui Integration
-----------------

Dear ImGui is widely used for debug UIs, tools, and editors. This integration provides a complete vuk-native renderer for ImGui that works seamlessly with the render graph system.

**Header**: ``vuk/extra/ImGuiIntegration.hpp``

**CMake Options**:

.. code-block:: cmake

   cmake_dependent_option(VUK_EXTRA_IMGUI "Build imgui integration" ON "VUK_EXTRA" OFF)
   set(VUK_EXTRA_IMGUI_PLATFORM_BACKEND "glfw" CACHE STRING "Platform backend")

Basic Usage
~~~~~~~~~~~

.. code-block:: cpp

   #include <vuk/extra/ImGuiIntegration.hpp>
   
   // Initialize once during setup
   vuk::extra::ImGuiData imgui_data = vuk::extra::ImGui_ImplVuk_Init(allocator);
   
   // Each frame, build your UI with standard ImGui calls
   ImGui::NewFrame();
   ImGui::ShowDemoWindow();
   ImGui::Render();
   
   // Render ImGui into your target
   auto with_ui = vuk::extra::ImGui_ImplVuk_Render(
       allocator,
       target_image,    // The image to render UI onto
       imgui_data       // Persistent ImGui data
   );

Using Custom Textures
~~~~~~~~~~~~~~~~~~~~~~

Display vuk images as ImGui textures:

.. code-block:: cpp

   // Add an image with custom sampler
   auto sampled_img = vuk::combine_image_sampler(
       "my_texture",
       my_image,
       vuk::acquire_sampler("nearest", {
           .magFilter = vuk::Filter::eNearest,
           .minFilter = vuk::Filter::eNearest
       })
   );
   ImTextureID tex_id = imgui_data.add_sampled_image(sampled_img);
   
   // Or use default sampler
   ImTextureID simple_id = imgui_data.add_image(my_image);
   
   // Use in ImGui
   ImGui::Image(tex_id, ImVec2(256, 256));

The integration automatically handles render graph dependencies and image transitions.

Simple Init
-----------

Setting up Vulkan and vuk requires significant boilerplate. These helpers streamline initialization for common use cases while maintaining flexibility for custom setups.

**Header**: ``vuk/extra/SimpleInit.hpp``
**CMake Options**:

.. code-block:: cmake

   cmake_dependent_option(VUK_EXTRA_INIT "Build init helper" ON "VUK_EXTRA" OFF)

**Dependencies**: Uses `vk-bootstrap <https://github.com/charles-lunarg/vk-bootstrap>`_ for Vulkan instance/device creation.

Quick Start Example with GLFW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <vuk/extra/SimpleInit.hpp>
   #include <GLFW/glfw.h>
   
   // Build instance
   auto instance_builder = vuk::extra::make_instance_builder(
       1, 3,    // Vulkan 1.3
       true     // Enable default debug callback
   );
   auto vkbinstance = instance_builder.build().value();
   
   // Create window with GLFW and create Vulkan surface
   // We use GLFW here for simplicity, but any surface creation method works
   glfwInit();
   glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
   GLFWwindow* window = glfwCreateWindow(1280, 720, "Vuk App", nullptr, nullptr);
   VkSurfaceKHR surface;
   glfwCreateWindowSurface(vkbinstance.instance, window, nullptr, &surface);
   
   // Select the physical device we want to use
   auto physical_device = vuk::extra::select_physical_device(
       vkbinstance,
       surface
   );
   
   // Create device and runtime
   vuk::extra::SimpleApp app = vuk::extra::make_device_builder(physical_device)
       .set_recommended_features()  // Enable common features
       .build_app(
           true,   // Create swapchain
           3       // 3 frames in flight
       );

The ``vuk::extra::SimpleApp`` class manages the complete Vulkan/vuk initialization and lifetime. It provides access to the instance, device, runtime, allocators, and swapchain, with methods for common operations like swapchain recreation and frame advancement.

.. doxygenstruct:: vuk::extra::SimpleApp
  :members:

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For custom device features:

.. code-block:: cpp

   auto device_builder = vuk::extra::make_device_builder(physical_device);
   
   // Access feature structures directly
   device_builder.vk12features.bufferDeviceAddress = true;
   device_builder.vk13features.dynamicRendering = true;
   
   // Or use recommended defaults
   device_builder.set_recommended_features();
   
   // Add custom pNext chains
   VkPhysicalDeviceCustomFeature custom_feature{};
   device_builder.add_pNext(&custom_feature);
   
   // Build device only (for manual runtime creation)
   auto device = device_builder.build_device_only().value();
   
   // Or build complete app
   auto app = device_builder.build_app();

SPD - Single Pass Downsampler
-----------------------------

Generating mipmaps is a common operation. AMD's SPD provides a highly optimized compute shader approach that generates all mip levels in a single pass, avoiding multiple dispatch overhead.

**Header**: ``vuk/extra/SPD.hpp``

**CMake**: Automatically included when ``VUK_EXTRA`` is enabled.

Usage
~~~~~

.. code-block:: cpp

   #include <vuk/extra/SPD.hpp>
   
   // Generate all mip levels of an image
   auto mipmapped_image = vuk::extra::generate_mips_spd(
       source_image,
       vuk::extra::ReductionType::Avg  // Averaging filter
   );

The following reduction types are supported:
.. doxygenenum:: vuk::extra::ReductionType

Complete Example
~~~~~~~~~~~~~~~~

.. code-block:: cpp

   // Create image with mip levels
   auto ia = vuk::ImageAttachment::from_preset(
       vuk::ImageAttachment::Preset::eGeneric2D,
       vuk::Format::eR8G8B8A8Srgb,
       vuk::Extent3D{ 512, 512, 1 },
       vuk::Samples::e1
   );
   ia.level_count = vuk::compute_mip_levels(ia.extent);
   
   auto [img, img_fut] = vuk::create_image_with_data(
       allocator,
       vuk::DomainFlagBits::eAny,
       ia,
       pixel_data
   );
   
   // Generate all mips
   img_fut = vuk::extra::generate_mips_spd(
       std::move(img_fut),
       vuk::extra::ReductionType::Avg
   );
   
   // Use the mipmapped image
   // All mip levels are now available for sampling

Constraints
~~~~~~~~~~~

- Maximum 13 mip levels supported (8192x8192 base resolution)
- Image must be square or rectangular power-of-two for optimal performance
- Works with any format, including sRGB (automatic handling)
- Requires compute shader support

Dependencies
------------

+------------------+--------------------------------+
| Feature          | External Dependencies          |
+==================+================================+
| ImGui            | Dear ImGui, platform backend   |
+------------------+--------------------------------+
| Simple Init      | vk-bootstrap                   |
+------------------+--------------------------------+
| SPD              | None (shader embedded)         |
+------------------+--------------------------------+

All dependencies are automatically fetched via CMake's FetchContent.