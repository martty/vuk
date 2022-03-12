Allocators
==========

Management of GPU resources is an important part of any renderer. vuk provides an API that lets you plug in your allocation schemes, complementing built-in general purpose schemes that get you started and give good performance out of the box.

Overview
--------

.. doxygenclass:: vuk::Allocator

.. doxygenstruct:: vuk::DeviceResource

To facilitate ownership, a RAII wrapper type is provided, that wraps an Allocator and a payload:

.. doxygenclass:: vuk::Unique

Built-in resources
------------------

.. doxygenstruct:: vuk::DeviceNestedResource

.. doxygenstruct:: vuk::DeviceVkResource

.. doxygenstruct:: vuk::DeviceFrameResource

.. doxygenstruct:: vuk::DeviceSuperFrameResource

Helpers
-------
Allocator provides functions that can perform bulk allocation (to reduce overhead for repeated calls) and return resources directly. However, usually it is more convenient to allocate a single resource and immediately put it into a RAII wrapper to prevent forgetting to deallocate it.

.. doxygenfile:: include/vuk/AllocatorHelpers.hpp


Reference
---------

.. doxygenclass:: vuk::Allocator
   :members:
