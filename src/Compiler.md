# VOIR and compiler explainer

## Features
- Partial lazy evaluation
- Inference of shapes
- Imperative effects

## Stages of compilation
1. Enumeration of participating IRModules & garbage collection
1. SSA conversion and linking of all IRModules into a single graph 
1. Build reachable set of nodes from tail nodes
1. Execute analysis and transformation passes on the graph
    1. Apply imperative effects
    1. Constant folding and shape inference
    1. Validation
        1. Duplicated resources
        1. Undefined reads
        1. Bad user passes
    1. Forced convergence
    1. Queue inference
    1. Computation of synchronization
1. Linearization of the graph into a sequence of operations
1. Execution
## Conceptual flow
### Node ponds into Sea-of-Nodes
In the initial stage, we have IRModules, which are the physical containers of nodes.
On the boundary of these modules, we might have references to nodes in other modules.


> Issue: How do we handle imperative code on module boundaries?

Say we have the following code:
```cpp
auto fut = something();
auto dst_buf = allocate...;
std::thread([&]() { copy(fut, std::move(dst_buf)); }).join();
... = download_buffer(dst_buf);
```
We have a `discard` node that creates a buffer, and then we have a `copy` node that writes to the buffer in another thread, and then we have a `download_buffer` node that reads from the buffer.
The `copy` node is in another module, and the `download_buffer` node is in the current module, but they are ordered - the `copy` must happen before the `download_buffer`.
However they are not connected in the graph, because the `copy` node does not produce a value that is consumed by the `download_buffer` node.

We now have two difficulties: we want to make sure both modules are alive, and we want to connect the nodes across the module boundary.

### SSA construction
> Issue: How do we handle imperative code?

Say we have the following code:
```cpp
auto my_image = allocate...;
copy(a_beautiful_image_of_mount_fuji, my_image.mip(0));
generate_mips(my_image);
```
In this code, we have a `copy` into the base mip level of `my_image`, and then we have a `generate_mips` that reads from the base mip level and writes to the other mip levels.
The `copy` and `generate_mips` nodes are not connected in the graph, because the `copy` node does not produce a value that is consumed by the `generate_mips` node.
Even if we had a `copy` node that produced a value, it would be a different value than the one consumed by the `generate_mips` node, because we want to pass the full image to `generate_mips`, not just the base mip level.
It is also infeasible to track this copy through changing `my_image`.
However, we want to ensure that the `copy` happens before the `generate_mips`, because otherwise we would be generating mips from uninitialized data.
Instead of requiring the user to manually connect these nodes, we want to automatically infer the dependency from the fact that they both access the same resource (`my_image`).

After linking and SSA construction, we have a single graph of nodes. At this point we can walk the graph from the tail nodes, and drop any nodes that are not reachable from the tail nodes.

### Shape inference and constant folding


### Execution and conversion into ACQUIRE
As we execute nodes, we compute their values and convert them into ACQUIRE operations. This is done for two reasons:
1. Any node must only be executed once - conversion to ACQUIRE removes the nodes's dependencies.
1. Subsequent compilations can refer to ACQUIRE operations instead of re-executing the nodes.


# C++ interface
To facilitate the construction of graphs, we have a C++ high-level interface that allows us to create nodes and connect them together, as if one was writing imperative C++ code.

Our code will look like normal C++ code, but under the hood we are actually constructing a graph of nodes. 
This creates a computation graph that is lazily evaluated when we need the result.

The core components here are `Value<T>`s, which represent values in the graph, and functions that take and return `Value<T>`s.

## Values
The `Value` class, represents a value in the graph.
When we create a `Value`, we are actually creating a node in the graph, and the `Value` object is a handle to that node.
When we pass a `Value` to a function, we are actually passing the node that the `Value` represents.
When we return a `Value` from a function, we are actually returning the node that the `Value` represents.
For example, consider the following code:
```cpp
Value<uint64_t> add(Value<uint64_t> a, Value<uint64_t> b) {
    return a + b;
}
Value<uint64_t> c = add(x, y);
```
In this code, we have a function `add` that takes two `Value` objects and returns a `Value` object. When we call `add(x, y)`, we are actually creating a new node in the graph that represents the addition of the nodes represented by `x` and `y`. The returned `Value` object `c` is a handle to this new node.

The `Value` class is templated on the type of the value it represents, which allows us to have type safety when constructing graphs.
The `Value` class also supports operator overloading, which allows us to use standard C++ operators to create nodes in the graph. For example, we can use the `+` operator to create an addition node, the `*` operator to create a multiplication node, and so on.

> What about user-defined types?

User-defined types can be used, as long as they can be represented in the IR.
This is done by defining a struct that represents the type, and then registering it with the compiler using a macro. For example:
```cpp
struct MyType {
    int a;
    float b;
};
ADAPT_STRUCT_FOR_IR(MyType, a, b);
```
You can then use `Value<MyType>` to represent a value of type `MyType` in the graph.
Alternatively, if you want to customize the representation of `MyType` in the IR, you can define a custom type by specializing the `Value<T>` template. For example:
```cpp
template<>
struct Value<MyType> {
    Value<int> a;
    Value<float> b;
};
```

> What does `MyType` vs. `Value<MyType>` mean?

`MyType` is the type you represent in the IR, while `Value<MyType>` is a handle while building the graph.
Alternatively you can say that the representation of `MyType` will be used in passes (by GPU and CPU operations inside passes), while `Value<MyType>` is the representation for the code that builds the graph.

## User-defined passes

As we can see from above, we can define functions that take and return `Value<T>`s, and use them to construct graphs.
But these functions happen while building the graph, not while executing it.
To define a function that will be executed as part of the graph, we need to define a pass.
The syntax for defining a pass looks like this:
```cpp
auto pass = vuk::make_pass("01_triangle", [](vuk::CommandBuffer& command_buffer, VUK_IA(vuk::eColorWrite) color_rt) {
	command_buffer.set_viewport(0, vuk::Rect2D::framebuffer());
	// Set the scissor area to cover the entire framebuffer
	command_buffer.set_scissor(0, vuk::Rect2D::framebuffer());
	command_buffer
		.set_rasterization({})              // Set the default rasterization state
		.set_color_blend(color_rt, {})      // Set the default color blend state
		.bind_graphics_pipeline("triangle") // Recall pipeline for "triangle" and bind
		.draw(3, 1, 0, 0);                  // Draw 3 vertices
	return color_rt;
});
```
Lets break this down:
1. We call `vuk::make_pass`, which takes a debug name and a lambda function.
1. The lambda function takes a `vuk::CommandBuffer&` and any number of arguments, and can return one of those arguments or a tuple of those arguments, in any order.
1. The arguments are annotated with vuk::Args, which specify how the argument is used in the pass. In this case, we have `VUK_IA(vuk::eColorWrite) color_rt`, which means that `color_rt` is an image that is used as a color attachment for writing.
1. Inside the lambda, we can use the `command_buffer` to record commands that will be executed when the pass is executed.
1. We can now call this pass like a normal function, passing in `Value<T>`s as arguments, and getting back `Value<T>`s as return values. For example:
```cpp
Value<ImageAttachment> color_image = ...;
Value<ImageAttachment> color_image_with_triangle = pass(color_image);
```

## Lazy evaluation
The graph is lazily evaluated, meaning that the nodes are not executed until we need the result or we explicitly request execution.
For example, consider the following code:
```cpp
Value<uint64_t> a = 1;
Value<uint64_t> b = 2;
Value<uint64_t> c = a + b;
// At this point, no nodes have been executed yet
uint64_t result = c.get(allocator, compiler); // Now the nodes are executed, and we get the result on the host
```

However, we can also explicitly request execution of the graph without getting the result on the host. This is done using the `submit` method:
```cpp
Value<uint64_t> a = 1;
Value<uint64_t> b = 2;
Value<uint64_t> c = a + b;
// At this point, no nodes have been executed yet
c.submit(allocator, compiler);
// Now the nodes are executed in the background (CPU and/or GPU)
// optionally we can wait for completion by doing
// c.wait(allocator, compiler); instead
```

## Basic operations

### acquire
The `acquire` operation is used to bring a resource into the graph. This can be a buffer, an image, a sampler, etc. The `acquire` operation takes a resource and returns a `Value<T>` that represents the resource in the graph. For example:
```cpp
vuk::Buffer buffer = ...; // A buffer created outside the graph
Value<vuk::Buffer> buffer_value = acquire(buffer);
```
### discard
The `discard` operation is used to create a new resource in the graph. This can be a buffer, an image, etc. The `discard` operation takes the type of the resource and its properties, and returns a `Value<T>` that represents the resource in the graph. For example:
```cpp
Value<vuk::Buffer> buffer_value = discard<vuk::Buffer>(vuk::BufferCreateInfo{ .size = 1024, .usage = vuk::eStorageBuffer });
```
### allocate
The `allocate` operation is used to allocate resource from an allocator. This can be a buffer, an image, etc. The `allocate` operation takes the type of the resource and its properties, and returns a `Value<T>` that represents the resource in the graph. For example:
```cpp
Value<vuk::Image> image_value = allocate<vuk::Image>(vuk::ImageCreateInfo{ .extent = { 512, 512, 1 }, .format = vuk::eR8G8B8A8Unorm, .usage = vuk::eSampled | vuk::eColorAttachment });
```
### acquire_next_image
The `acquire_next_image` operation is used to acquire the next image from a swapchain. This is typically used in rendering applications where we need to render to the next available image in the swapchain. For example:
```cpp
Value<vuk::Image> swapchain_image = acquire_next_image(swapchain);
```
### enqueue_presentation
The `enqueue_presentation` operation is used to present an image to the swapchain. This is typically used in rendering applications where we need to present the rendered image to the screen. For example:
```cpp  
enqueue_presentation(swapchain, rendered_image);
```
### compile_pipeline
The `compile_pipeline` operation is used to compile a graphics or compute pipeline. 
This can be useful when you want to use pipelines inside functions, perhaps as glue code or to build higher-level abstractions.
This operation takes the pipeline configuration and returns a `Value<T>` that represents the compiled pipeline in the graph. For example:
```cpp
Value<PipelineBaseInfo*> pipeline = compile_pipeline(vuk::PipelineCreateInfo{ ... });
```
This Value can then be used in passes to bind the pipeline for rendering or compute operations.
For example, in a pass:
```cpp
auto pass = vuk::make_pass("pass", [](vuk::CommandBuffer& command_buffer, PipelineBaseInfo* pipeline) {
    command_buffer.bind_graphics_pipeline(pipeline);
    // Other rendering commands...
});
```

## Common types
We have a number of common types that are used in the graph, such as `Buffer`, `Image`, `Sampler`, `Pipeline`, etc.

### Arrays

### Unions


## Standard library functions
We have a number of standard library functions that can be used in the graph, such as `min`, `max`, `clamp`, `lerp`, etc.

# Vuk user guide
Vuk is a modern C++ graphics API that aims to make it easy to write high-performance graphics applications. 
It has a Vulkan backend, and provides a high-level interface for common graphics tasks.