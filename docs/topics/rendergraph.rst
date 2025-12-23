Core concepts
=============

Vuk lets you write normal imperative code where :cpp:class:`vuk::Value` types stand in for ordinary variables, and passes created with :cpp:func:`vuk::make_pass` behave like regular functions. The key difference is that execution is **lazy** - work is deferred until you explicitly observe a result.

**Think of it like this:**

- :cpp:class:`vuk::Value\<T>` = like a variable of type T
- :cpp:func:`vuk::make_pass` = like a defining a function
- Function calls chain together like normal C++ code
- Nothing actually executes until requested or required

**Example - looks like imperative code:**

.. code-block:: cpp

   // These look like normal variable declarations and function calls
   auto uploaded_data = upload_vertices(allocator, vertex_data);
   auto cleared_image = clear_pass(render_target);
   auto rendered_image = draw_pass(uploaded_data, cleared_image);
   auto post_processed = blur_pass(rendered_image);
   
   // But nothing has executed yet! The GPU work is lazily evaluated.
   // Only when we observe the result does execution happen:
   post_processed.wait(allocator, compiler);  // Now everything runs

The idea is that you write straightforward, easy-to-read code, while vuk automatically take care of translating it for the underlying graphics API.
When you need to actually use a result on the CPU, you force the computations to happen. When you need a result on the GPU, you just make new passes that use the result, and vuk figures out the dependencies.
This lazy evaluation model means you can build complex render graphs using familiar programming patterns, without worrying about the underlying complexity of GPU synchronization and resource management.

Values
------

At the heart of vuk's execution model is the :cpp:class:`vuk::Value` type. A :cpp:class:`vuk::Value` represents a resource (like an image or buffer, or even an integer) that will be available after some GPU work completes.

.. code-block:: cpp

   // Create a buffer with data - returns the buffer handle and a Value
   auto [buf_handle, buffer_value] = create_buffer(
       allocator, 
       MemoryUsage::eGPUonly, 
       DomainFlagBits::eTransferOnTransfer, 
       std::span(my_data)
   );
   
   // buffer_value is a Value<Buffer> representing the buffer after upload completes
   // The actual upload hasn't happened yet!

:cpp:class:`vuk::Value` types are composable - you can chain operations on them to build complex pipelines without explicitly managing dependencies or synchronization.

Building GPU computation with make_pass
---------------------------------------

The :cpp:func:`vuk::make_pass` function creates a render graph node that performs some GPU work. It automatically infers dependencies and synchronization from the resources you use.
The first argument is a name for the pass (for debugging), followed by a lambda that records commands into a :cpp:class:`vuk::CommandBuffer`. The lambda's parameters specify the resources it uses, annotated with access patterns.
The lambda must always take a ``vuk::CommandBuffer&`` as the first parameter, followed by any number of resources annotated with access patterns. The lambda returns the output resources.
If the lambda returns multiple resources, return them as a :cpp:struct:`std::tuple`.

After making the pass, you can call it like a normal function, passing in :cpp:class:`vuk::Value` s representing the input resources. The result is a :cpp:class:`vuk::Value` representing the output resource(s) or a tuple of :cpp:class:`vuk::Value` s.

.. note::
   Buffers and ImageAttachments can be mutated - imagine as if the function takes them by reference and modifies them in place.

For now we are focusing on how to create the whole program - see the section on CommandBuffer for details on what goes into the callback.

Basic Structure
^^^^^^^^^^^^^^^

.. code-block:: cpp

   auto my_pass = make_pass(
       "pass_name",
       [](CommandBuffer& cbuf, VUK_IA(eColorWrite) target) {
           // Record commands into cbuf
           cbuf.draw(3, 1, 0, 0);
           
           // Return the output resource
           return target;
       }
   );

We have to annotate resource parameters with access patterns so vuk can manage synchronization. Use these macros for convenience:

- ``VUK_IA(access)`` - :cpp:struct:`vuk::ImageAttachment` with specified access pattern
- ``VUK_BA(access)`` - :cpp:struct:`vuk::Buffer` with specified access pattern
- ``VUK_ARG(type, access)`` - Generic argument with access pattern

.. warning::
   While it is possible to capture resources into the lambda via captures (e.g., ``[my_buffer]``), doing so bypasses vuk's automatic synchronization. When you capture resources directly, **you become responsible for managing synchronization manually**. It is recommended to pass all resources as lambda parameters with proper access annotations instead, allowing vuk to handle synchronization automatically.

Building up a program
---------------------

We have seen that Values represent resources that will be available after GPU work, and that :cpp:func:`vuk::make_pass` creates functions that operate on these resources. By combining these two concepts, we can build complex GPU computations in a straightforward way.
First, we have to make the input resources available as :cpp:class:`vuk::Value` s. This is done via resource declaration and acquisition functions. 
After that, we can chain together passes using normal function call syntax. Finally, we force evaluation.

Resource Declaration and Acquisition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Vuk provides several functions to create and import resources:

- **declare_ia / declare_buf** - Create placeholder resources that will be allocated later
- **acquire_ia / acquire_buf** - Import existing resources (e.g., from previous frames)
- **discard_ia / discard_buf** - Specify that previous contents don't matter (will overwrite)

.. note::
    The first argument to these functions is a debug name for the resource.

.. code-block:: cpp

   // Declare an image that will be allocated later
   auto temp_image = declare_ia("temp", ImageAttachment{
       .format = Format::eR8G8B8A8Srgb,
       .extent = {1024, 768, 1}
   });
   
   // Import an existing image (e.g., from a previous frame)
   auto imported = acquire_ia(
       "imported",
       existing_image_attachment,
       Access::eFragmentSampled  // Last known access
   );
   
   // We don't care about previous contents, will overwrite
   auto fresh_target = discard_ia("target", render_target);

**Resources**

:cpp:struct:`vuk::Buffer` and :cpp:struct:`vuk::ImageAttachment` are the fundamental resource types in vuk:

- **Buffer** - Represents GPU memory for storing arbitrary data (vertices, indices, uniforms, storage buffers). Contains information about size, memory usage, and optional device address for buffer device address features.

- **ImageAttachment** - Represents a GPU image/texture with specified format, dimensions, sample count, and mip/array layer configuration. Used for render targets, textures, and framebuffer attachments.

Both types are handles to memory that can be copied freely. The actual GPU resources are managed by allocators.

**Letting vuk allocate for you**

When you use :cpp:func:`vuk::declare_ia` or :cpp:func:`vuk::declare_buf`, you're creating resources without allocating actual GPU upfront. vuk will automatically allocate the necessary memory when the render graph executes, based on how the resources are used in your passes.
This is great for transient resources that only exist within a single frame or render pass. 
For resources that need to persist across frames (like geometry buffers or persistent textures), it can be more convenient to create these upfront and then import with :cpp:func:`vuk::acquire_ia` / :cpp:func:`vuk::acquire_buf`.

.. code-block:: cpp

   // Transient resources - let vuk allocate and manage
   auto temp_rt = declare_ia("temp_render_target", ImageAttachment{
       .format = Format::eR16G16B16A16Sfloat,
       .extent = {1920, 1080, 1}
   });
   
   // During startup:
   // Persistent resources - allocate once, reuse across frames
   auto [persistent_buf, upload_future] = create_buffer(
       allocator,
       MemoryUsage::eGPUonly,
       DomainFlagBits::eTransferOnGraphics,
       geometry_data
   );
   upload_future.wait(allocator, compiler);  // Ensure upload completes before use
   
   // During frame rendering:
   auto persistent_buf_val = acquire_buf(
           "persistent_geometry",
           persistent_buf,
           Access::eTransferWrite  // Last known access
   );

Building up complex computations from passes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passes created with :cpp:func:`vuk::make_pass` can be chained together like normal functions. Each pass takes :cpp:class:`vuk::Value` s as inputs and produces :cpp:class:`vuk::Value` s as outputs.

.. code-block:: cpp

   // Each pass takes inputs and produces outputs
   auto pass1 = make_pass("clear", 
       [](CommandBuffer& cbuf, VUK_IA(eColorWrite) img) {
           cbuf.clear_image(img, ClearColor{0.f, 0.f, 0.f, 1.f});
           return img;
       });
   
   auto pass2 = make_pass("draw", 
       [](CommandBuffer& cbuf, VUK_IA(eColorWrite) img) {
           cbuf.bind_graphics_pipeline("my_pipeline");
           cbuf.draw(3, 1, 0, 0);
           return img;
       });
   
   // Chain them together
   Value<ImageAttachment> result = pass2(pass1(my_image));


Presentation
^^^^^^^^^^^^

To display rendered images on screen, use the swapchain functions to acquire images, render to them, and present:

.. code-block:: cpp

   // Acquire the swapchain as a Value
   auto swapchain_val = acquire_swapchain(my_swapchain);
   
   // Get the next image from the swapchain
   auto swapchain_image = acquire_next_image("swapchain_img", swapchain_val);

   // additional rendering passes...   
   // Render to the swapchain image
   auto rendered = render_pass(swapchain_image);
   // additional rendering passes...
   
   // Mark the image ready for presentation
   auto presentable = enqueue_presentation(rendered);

Execution
^^^^^^^^^

Once you have built up your computation using passes and values, you need to trigger execution.
:cpp:class:`vuk::Value\<T>` provides several methods to control when work executes:

- **submit()** - Queue work for execution without waiting (non-blocking)
- **wait()** - Submit and wait for completion (blocking)
- **get()** - Submit, wait, and retrieve the result (blocking with data retrieval)

.. code-block:: cpp

   // Build the graph
   auto result = my_pass(input_image);
   
   // Option 1: Submit without waiting (non-blocking)
   result.submit(allocator, compiler);
   // Do other work while GPU executes...
   
   // Option 2: Submit and wait for completion
   result.wait(allocator, compiler);
   
   // Option 3: For CPU readback - submit, wait, and retrieve
   auto final_buffer = download_buffer(gpu_buffer);
   auto cpu_result = final_buffer.get(allocator, compiler);
   auto data = std::span((uint32_t*)cpu_result->mapped_ptr, element_count);

In all cases, computation only happens once. Subsequent calls to ``submit()``, ``wait()``, or ``get()`` on the same :cpp:class:`vuk::Value` do not re-execute the work; they simply ensure the result is ready.

.. note::
   You don't need to wait if you are only using the result on the GPU - just pass the :cpp:class:`vuk::Value` to another pass and vuk will handle dependencies automatically.
   You can also submit, then use that value in other passes - this means the computation for that intermediate result will be scheduled independently.

.. warning::
   Calling ``get()`` incurs CPU-GPU synchronization and should be avoided in throughput-critical paths. Prefer chaining passes when possible.

.. warning::
   vuk can only see the computations until the point you call an execution method. If you rely on vuk determining eg. image usage, be sure that vuk can see every use or make the image yourself.


Advanced topics
-----------------

Resources outside the render graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

vuk can only reason about resources that are part of the render graph. If you want to use a resource outside the graph, you can specify the desired final state using :cpp:func:`vuk::Value::as_released`.

.. code-block:: cpp

   auto released = my_value.as_released(
       Access::eFragmentSampled,  // Future access outside the graph
       DomainFlagBits::eGraphicsQueue  // Queue that will use it
   );

Conversely, if you have a resource created outside the graph that you want to use inside, use :cpp:func:`vuk::acquire_ia` / :cpp:func:`vuk::acquire_buf` to import it with its last known access pattern.

.. code-block:: cpp

   auto imported = acquire_buf(
       "imported_buf",
       existing_buffer,
       Access::eTransferWrite  // Last known access
   );

Multi-queue execution
^^^^^^^^^^^^^^^^^^^^^

Vuk automatically schedules work across multiple queues:

.. code-block:: cpp

   auto transfer_pass = make_pass("upload", 
       [](CommandBuffer& cbuf, VUK_BA(eTransferWrite) dst) {
           cbuf.fill_buffer(dst, 0);
           return dst;
       },
       DomainFlagBits::eTransferQueue  // Schedule on transfer queue
   );

Resource inference
^^^^^^^^^^^^^^^^^^
Vuk can infer resource properties like sizes and formats from other resources in the graph. 
This is particularly useful when you have resources whose dimensions or properties depend on other resources, but you don't want to manually track these dependencies.

**Inference methods available:**

For :cpp:class:`vuk::Value\<ImageAttachment>`:

- ``same_extent_as(src)`` - Infer width, height, and depth
- ``same_2D_extent_as(src)`` - Infer width and height only
- ``same_format_as(src)`` - Infer format
- ``same_shape_as(src)`` - Infer extent, layers, and mip levels
- ``similar_to(src)`` - Infer all properties (shape, format, sample count)

For :cpp:class:`vuk::Value\<Buffer>`:

- ``same_size(src)`` - Infer buffer size
- ``get_size()`` - Get size as a ``Value<uint64_t>`` for computation
- ``set_size(size_value)`` - Set size from a computed value

**How built-in functions use inference:**

Many of vuk's built-in functions automatically set up inference for you. For example, when copying between images:

.. code-block:: cpp

   auto src = acquire_ia("source", source_image, Access::eTransferRead);
   auto dst = declare_ia("destination");  // No properties specified
   
   // copy() automatically sets up inference
   auto copied = copy(src, dst);
   // dst now has the same extent as src

Similarly, ``download_buffer`` infers the required buffer size:

.. code-block:: cpp

   auto gpu_image = render_pass(target);
   
   // download_buffer automatically creates a buffer with the right size
   // based on the image format and dimensions
   auto cpu_buffer = download_buffer(gpu_image);
   auto result = cpu_buffer.get(allocator, compiler);
   
   // result->size is automatically set to the image's data size

**Practical example - Blur pipeline:**

Here's a complete example showing how inference simplifies a blur pipeline:

.. code-block:: cpp

   // Input image with some specific properties
   auto input = acquire_ia("input", source_image, Access::eFragmentSampled);
   
   // Horizontal blur - needs same size as input
   auto h_blur = declare_ia("horizontal_blur");
   h_blur.similar_to(input);  // Automatically matches all properties
   
   // Vertical blur - needs same size as horizontal
   auto v_blur = declare_ia("vertical_blur");
   v_blur.similar_to(h_blur);  // Chain inference
   
   // Perform the blur
   auto h_blurred = horizontal_blur_pass(input, h_blur);
   auto final = vertical_blur_pass(h_blurred, v_blur);
   
   // If input changes size, everything adapts automatically!

**Computing with inferred values:**

You can also perform arithmetic on inferred values:

.. code-block:: cpp

   auto input_buf = acquire_buf("input", input_buffer, Access::eTransferRead);
   
   // Create output buffer that's twice the size
   auto output_buf = declare_buf("output");
   output_buf->memory_usage = MemoryUsage::eGPUonly;
   output_buf.set_size(input_buf.get_size() * 2);  // Arithmetic on sizes!
   
   // Create another buffer based on image dimensions
   auto image = acquire_ia("img", my_image, Access::eFragmentSampled);
   auto pixel_buffer = declare_buf("pixels");
   pixel_buffer->memory_usage = MemoryUsage::eGPUonly;
   
   // Compute size from image dimensions
   auto width = image.get_size().get_width();
   auto height = image.get_size().get_height();
   auto pixel_count = width * height;
   pixel_buffer.set_size(pixel_count * 4);  // 4 bytes per pixel (RGBA8)

This approach makes your render graphs more maintainable - when input dimensions change, everything downstream adapts automatically without code changes.

Domains and Access
==================

Domains: Where Work Happens
----------------------------

A :cpp:enum:`vuk::DomainFlagBits` specifies *where* work should execute. GPUs have different queues specialized for different types of work:

- ``eGraphicsQueue`` - Graphics with rasterization, compute shaders, and transfer
- ``eComputeQueue`` - Compute shaders and transfer
- ``eTransferQueue`` - Memory transfers (uploads/downloads)
- ``eHost`` - CPU-side operations
- ``eAny`` - Let vuk decide

.. code-block:: cpp

   // This upload can happen in parallel with other graphics work
   auto transfer_pass = make_pass("upload", 
       [](CommandBuffer& cbuf, VUK_BA(eTransferWrite) dst) {
           cbuf.fill_buffer(dst, 0);
           return dst;
       },
       DomainFlagBits::eTransferQueue  // Execute on transfer queue
   );

Most of the time you can use ``DomainFlagBits::eAny`` and let vuk infer the queue.

Access: How Resources Are Used
-------------------------------

An :cpp:enum:`vuk::Access` pattern tells vuk *how* a resource will be used in a pass. This is critical for:

1. **Synchronization** - Ensuring reads happen after writes
2. **Layout transitions** - Images need different layouts for different operations
3. **Cache management** - Proper invalidation and flushing

.. doxygenenum:: vuk::Access

Example showing different access patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: cpp

   // Clear an image (write access)
   auto cleared = make_pass("clear",
       [](CommandBuffer& cbuf, VUK_IA(eColorWrite) target) {
           cbuf.clear_image(target, ClearColor{0.f, 0.f, 0.f, 1.f});
           return target;
       })(my_image);
   
   // Read it as a texture (read access)
   auto sampled = make_pass("sample",
       [](CommandBuffer& cbuf, VUK_IA(eFragmentSampled) texture) {
           // Use texture in shader
           return texture;
       })(cleared);

Vuk uses these access patterns to automatically:

- Insert memory barriers
- Transition image layouts (e.g., ``TRANSFER_DST`` → ``SHADER_READ_ONLY``)
- Determine execution dependencies
- Order passes correctly


API reference for Value and make_pass
-------------------------------------

.. doxygenclass:: vuk::Value
   :members:

.. doxygenfunction:: vuk::make_pass