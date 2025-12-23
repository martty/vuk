Advanced Topics
===============

.. note::
   This section is for advanced users.

Streams: Sequences of GPU Work
-------------------------------

A :cpp:class:`vuk::Stream` represents a sequence of GPU operations that execute in order on a particular queue. When you build a render graph, vuk internally creates streams to organize work.

**Key concepts:**

- Each stream corresponds to a command buffer being recorded
- Operations within a stream are guaranteed to execute in order
- Operations in different streams can execute in parallel
- Vuk automatically manages stream creation and synchronization

**When streams are created:**

- When work needs to move to a different queue (graphics ? transfer ? compute)
- When explicit synchronization is needed (via ``as_released()``)
- When the compiler determines parallel execution is beneficial

You typically don't create streams directly - vuk creates them as needed when compiling your render graph.

Executors: Queue Abstractions
------------------------------

An :cpp:class:`vuk::Executor` is vuk's abstraction over Vulkan queues. When you create a :cpp:class:`vuk::Runtime`, you provide executors for the queues you want to use:

.. code-block:: cpp

   // Executors are typically created during Runtime initialization
   RuntimeCreateParameters params;
   params.executors.push_back(
       create_vkqueue_executor(fps, device, graphics_queue, 
                               graphics_family_index, 
                               DomainFlagBits::eGraphicsQueue)
   );
   params.executors.push_back(
       create_vkqueue_executor(fps, device, transfer_queue, 
                               transfer_family_index, 
                               DomainFlagBits::eTransferQueue)
   );

**Executor responsibilities:**

- Submitting command buffers to Vulkan queues
- Managing queue submission order
- Handling semaphore synchronization between queues
- Tracking submission completion

**When you might interact with executors:**

- Custom queue configurations (e.g., multiple compute queues)
- Profiling and debugging (inspecting which executor ran what work)
- Advanced multi-queue scenarios

For most applications, the executors created during initialization are sufficient, and vuk handles all executor management automatically.

Stream and Executor Interaction
--------------------------------

Here's how these concepts work together:

1. You build a render graph with ``make_pass`` and ``Value``
2. When you call ``wait()`` or ``get()``, vuk compiles the graph
3. The compiler creates **streams** for different queues
4. Each stream records commands into a command buffer
5. Streams are submitted to **executors**
6. Executors submit to Vulkan queues with proper synchronization

.. code-block:: cpp

   // This single pipeline might use multiple streams and executors:
   auto uploaded = upload_data(allocator, data);        // Transfer executor
   auto processed = compute_pass(uploaded);             // Compute executor  
   auto rendered = render_pass(processed, target);      // Graphics executor
   
   rendered.wait(allocator, compiler);
   // Vuk coordinates all three executors with proper synchronization
