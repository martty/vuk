![vuk logo](https://github.com/martty/vuk/blob/master/vuk_logo.png)

### **vuk** - A rendergraph-based abstraction for Vulkan

[![Discord Server](https://img.shields.io/discord/939539624039186432?style=for-the-badge)](https://discord.gg/UNkJMHgUmZ)
[![Documentation](https://img.shields.io/readthedocs/vuk/v0.3?style=for-the-badge)](https://vuk.readthedocs.io/en/v0.3/)

### Quick Start
1. Grab the vuk repository
2. Compile the examples
3. Run the example browser and get a feel for the library
```
git clone http://github.com/martty/vuk
cd vuk
git submodule init
git submodule update --recursive
mkdir build
cd build
mkdir debug
cd debug
cmake ../.. -G Ninja -DCMAKE_BUILD_TYPE=Debug -DVUK_BUILD_EXAMPLES=ON # prefix with CC=clang CXX=clang++ if your default compiler isn't Clang or M$VC
cmake --build .
./vuk_all_examples
```
(if building with a multi-config generator, do not make the `debug` folder)

### Overview of using **vuk**
1. Initialize your window(s) and Vulkan device
2. Create a `vuk::Context` object
3. Each frame:
  1. Each frame, prepare high level description of your rendering, in the form of `vuk::Pass`
  2. Bind concrete resources as inputs and outputs
  3. Bind managed resources (temporary resources used by the rendergraph)
  4. Record the execution your rendergraph into a command buffer
  5. Submit and present

### What does **vuk** do
- [x] Automatically deduces renderpasses, subpasses and framebuffers
  - [x] with all the synchronization handled for you
   - [x] including buffers
   - [x] images
   - [x] and rendertargets.
  - [x] for multiple queues
  - [ ] using fine grained synchronization when possible (events)
- [x] Automatically transitions images into proper layouts
  - [x] for renderpasses
  - [x] and commands outside of renderpasses (eg. blitting).
- [x] Automates pipeline creation with
  - [x] optionally compiling your shaders at runtime using shaderc
  - [x] pipeline layouts and
  - [x] descriptor set layouts
  - [x] by reflecting your shaders
  - [x] and deducing parameters based on renderpass and framebuffer.
- [x] Automates resource binding with hashmaps, reducing descriptor set allocations and updates.
- [x] Handles temporary allocations for a frame
- [x] Handles long-term allocations with RAII handles
- [x] Comes with lots of sugar to simplify common operations, but still exposing the full Vulkan interface:
  - [x] Matching viewport/scissor dimensions to attachment sizes
  - [x] Simplified vertex format specification
  - [x] Blend presets
  - [x] Directly writable mapped UBOs
  - [x] Automatic management of multisampling
- [x] Helps debugging by naming the internal resources
- [x] dear imgui integration code
