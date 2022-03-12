Rendergraph
===========

.. doxygenstruct:: vuk::Resource
  :members:

.. doxygenstruct:: vuk::RenderGraph
  :members:

.. doxygenstruct:: vuk::ExecutableRenderGraph
  :members:

Futures
=======
vuk Futures allow you to reason about computation of resources that happened in the past, or will happen in the future. In general the limitation of RenderGraphs are that they don't know the state of the resources produces by previous computation, or the state the resources should be left in for future computation, so these states must be provided manually (this is error-prone). Instead you can encapsulate the computation and its result into a Future, which can then serve as an input to other RenderGraphs.

Futures can be constructed from a RenderGraph and a named Resource that is considered to be the output. A Future can optionally own the RenderGraph - but in all cases a Future must outlive the RenderGraph it references.

You can submit Futures manually, which will compile, execute and submit the RenderGraph it references. In this case when you use this Future as input to another RenderGraph it will wait for the result on the device. If a Future has not yet been submitted, the contained RenderGraph is simply appended as a subgraph (i.e. inlined).

It is also possible to wait for the result to be produced to be available on the host - but this forces a CPU-GPU sync and should be used sparingly.

.. doxygenclass:: vuk::Future
  :members:
  
Composing render graphs
=======================
Futures make easy to compose complex operations and effects out of RenderGraph building blocks, linked by Futures. These building blocks are termed partials, and vuk provides some built-in. Such partials are functions that take a number of Futures as input, and produce a Future as output.

The built-in partials can be found below. Built on these, there are some convenience functions that couple resource allocation with initial data (`create_XXX()`).

.. doxygenfile:: include/vuk/Partials.hpp