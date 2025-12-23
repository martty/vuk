Runtime
=======
The Runtime represents the base object of the runtime, encapsulating the knowledge about the GPU (similar to a :cpp:struct:`VkDevice`).
Use this class to manage pipelines and other cached objects, add/remove swapchains, manage persistent descriptor sets, submit work to device and retrieve query results.


.. doxygenstruct:: vuk::RuntimeCreateParameters
    :members:

.. doxygenclass:: vuk::Runtime
    :members:
    

.. doxygenstruct:: vuk::Query