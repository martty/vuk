CommandBuffer
=============

The CommandBuffer class offers a convenient abstraction over command recording, pipeline state and descriptor sets of Vulkan.

Setting pipeline state
----------------------
The CommandBuffer encapsulates the current pipeline and descriptor state. When calling state-setting commands, the current state of the CommandBuffer is updated. The state of the CommandBuffer persists for the duration of the execution callback, and there is no state leakage between callbacks of different passes.

The various states of the pipeline can be reconfigured by calling the appropriate function, such as :cpp:func:`vuk::CommandBuffer::set_rasterization()`.

There is no default state - you must explicitly bind all state used for the commands recorded.

Static and dynamic state
------------------------
Vulkan allows some pipeline state to be dynamic. In vuk this is exposed as an optimisation - you may let the CommandBuffer know that certain pipeline state is dynamic by calling :cpp:func:`vuk::CommandBuffer::set_dynamic_state()`. This call changes which states are considered dynamic. Dynamic state is usually cheaper to change than entire pipelines and leads to fewer pipeline compilations, but has more overhead compared to static state - use it when a state changes often. Some state can be set dynamic on some platforms without cost. As with other pipeline state, setting states to be dynamic or static persist only during the callback.

Binding pipelines & specialization constants
--------------------------------------------
The CommandBuffer maintains separate bind points for compute and graphics pipelines. The CommandBuffer also maintains an internal buffer of specialization constants that are applied to the pipeline bound. Changing specialization constants will trigger a pipeline compilation when using the pipeline for the first time.

Binding descriptors & push constants
------------------------------------
vuk allows two types of descriptors to be bound: ephemeral and persistent. 

Ephemeral descriptors are bound individually to the CommandBuffer via `bind_XXX()` calls where `XXX` denotes the type of the descriptor (eg. uniform buffer). These descriptors are internally managed by the CommandBuffer and the Allocator it references. Ephemeral descriptors are very convenient to use, but they are limited in the number of bindable descriptors (`VUK_MAX_BINDINGS`) and they incur a small overhead on bind.

Persistent descriptors are managed by the user via allocation of a PersistentDescriptorSet from Allocator and manually updating the contents. There is no limit on the number of descriptors and binding such descriptor sets do not have an overhead over the direct Vulkan call. Large descriptor arrays (such as the ones used in "bindless" techniques) are only possible via persistent descriptor sets.

The number of bindable sets is limited by `VUK_MAX_SETS`. Both ephemeral descriptors and persistent descriptor sets retain their bindings until overwritten, disturbed or the the callback ends.

Push constants can be changed by calling :cpp:func:`vuk::CommandBuffer::push_constants()`.

Command recording
-----------------
Draws and dispatches can be recorded by calling the appropriate function. Any state changes made will be recorded into the underlying Vulkan command buffer, along with the draw or dispatch.

Error handling
--------------
The CommandBuffer implements "monadic" error handling, because operations that allocate resources might fail. In this case the CommandBuffer is moved into the error state and subsequent calls do not modify the underlying state.

.. doxygenclass:: vuk::CommandBuffer
   :members: