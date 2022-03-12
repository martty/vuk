.. vuk documentation master file, created by
   sphinx-quickstart on Thu Dec  3 19:06:20 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to vuk's documentation!
===============================

Quickstart
==========
1. Grab the vuk repository
2. Compile the examples
3. Run the example browser and get a feel for the library::

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

(if building with a multi-config generator, do not make the `debug` folder)

.. toctree::
   :maxdepth: 2
   :caption: Topics:
   
   topics/context
   topics/allocators
   topics/rendergraph
   topics/commandbuffer


Background
==========
vuk was initially conceived based on the rendergraph articles of themaister (https://themaister.net/blog/2017/08/15/render-graphs-and-vulkan-a-deep-dive/). In essence the idea is to describe work undertaken during a frame in advance in a high level manner, then the library takes care of low-level details, such as insertion of synchronization (barriers) and managing resource states (image layouts). This over time evolved to a somewhat complete Vulkan runtime - you can use the facilities afforded by vuk's runtime without even using the rendergraph part. The runtime presents a more easily approachable interface to Vulkan, abstracting over common pain points of pipeline management, state setting and descriptors. The rendergraph part has grown to become more powerful than simple 'autosync' abstraction - it allows expressing complex dependencies via `vuk::Future` and allows powerful optimisation opportunities for the backend (even if those are to be implemented).

Alltogether vuk presents a vision of GPU development that embraces compilation - the idea that knowledge about optimisation of programs can be encoded into to tools (compilers) and this way can be insitutionalised, which allows a broader range of programs and programmers to take advantage of these. The future developments will focus on this backend(Vulkan, DX12, etc.)-agnostic form of representing graphics programs and their optimisation.

As such vuk is in active development, and will change in API and behaviour as we better understand the shape of the problem. With that being said, vuk is already usable to base projects off of - with the occasional refactoring. For support or feedback, please join the Discord server or use Github issues - we would be very happy to hear your thoughts!

Indices and tables
==================

* :ref:`genindex`
