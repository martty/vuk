Context
=======
The Context represents the base object of the runtime, encapsulating the knowledge about the GPU (similar to a VkDevice).
Use this class to manage pipelines and other cached objects, add/remove swapchains, manage persistent descriptor sets, submit work to device and retrieve query results.


.. doxygenstruct:: vuk::ContextCreateParameters
    :members:

.. doxygenclass:: vuk::Context
    :members:
    

.. doxygenstruct:: vuk::Query

Submitting work
===============
While submitting work to the device can be performed by the user, it is usually sufficient to use a utility function that takes care of translating a RenderGraph into device execution. Note that these functions are used internally when using :cpp:class:`vuk::Future`s, and as such Futures can be used to manage submission in a more high-level fashion.

.. doxygenfunction:: vuk::execute_submit_and_present_to_one

.. doxygenfunction:: vuk::execute_submit_and_wait

.. doxygenfunction:: vuk::link_execute_submit
