Allocators
==========

Management of GPU resources is an important part of any renderer. vuk provides an API that lets you plug in your allocation schemes, complementing built-in general purpose schemes that get you started and give good performance.

Overview
--------

.. doxygenclass:: vuk::Allocator

.. doxygenstruct:: vuk::DeviceResource


Built-in resources
------------------

.. doxygenstruct:: vuk::DeviceNestedResource

.. doxygenstruct:: vuk::DeviceVkResource

.. doxygenstruct:: vuk::DeviceFrameResource

.. doxygenstruct:: vuk::DeviceSuperFrameResource

Helpers
-------

.. doxygenfile:: include/vuk/AllocatorHelpers.hpp


Reference
---------

.. doxygenclass:: vuk::Allocator
   :members:
