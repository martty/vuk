### **vuk** - A rendergraph-based abstraction for Vulkan

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
cmake ../.. -G Ninja
cmake --build .
./vuk_all_examples
```
(if building with a multi-config generator, do not make the `debug` folder)

### Overview of using **vuk**
3. Initialize your window(s) and Vulkan device
3. Create a `vuk::Context` object
4. Each frame:
  1. Each frame, prepare high level description of your rendering, in the form of `vuk::Pass`
  2. Bind concrete resources as inputs and outputs
  3. Bind temporary resources
  4. Record the execution your rendergraph into `vk::CommandBuffer`(s)
  5. Submit and present

### What does **vuk** do
- [x] Automatically deduces renderpasses, subpasses and framebuffers
  - [ ] in an optimal way
  - [x] with all the synchronization handled for you
   - [ ] including buffers
   - [x] images
   - [x] and rendertargets.
- [x] Automatically transitions images into proper layouts
  - [x] for renderpasses
  - [ ] and commands outside of renderpasses (eg. blitting).
- [x] Automates pipeline creation with
  - [x] pipeline layouts and
  - [x] descriptor set layouts
  - [x] by reflecting your shaders
  - [x] and deducing parameters based on renderpass and framebuffer.
- [x] Automates resource binding with hashmaps, reducing descriptor set allocations and updates.
- [x] Handles temporary allocations for a frame
- [ ] Handles long-term allocations with RAII handles
  - [x] for images
  - [ ] and buffers.
- [x] Comes with lots of sugar to simplify common operations, but still exposing the full Vulkan interface:
  - [x] Matching viewport/scissor dimensions to attachment sizes
  - [x] Simplified vertex format specification
  - [x] Blend presets
  - [x] Directly writable mapped UBOs
  - [x] Automatic management of multisampling
- [x] Helps debugging by naming the internal resources
- [x] dear imgui integration code
- [ ] Error checking
