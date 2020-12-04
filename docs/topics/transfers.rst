Transfers
=========
vuk exposes two main methods of uploading data to the GPU: fenced uploads and frame uploads. Fenced uploads execute 
independently of the frames-in-flight. It is the application's responsibility to poll for completion of fenced uploads.

Fenced uploads can be used to upload to either :cpp:struct:`buffers <vuk::Buffer>`:

.. doxygenstruct:: vuk::Context::BufferUpload
   :members:

or Images :

.. doxygenstruct:: vuk::Context::ImageUpload
   :members:

The upload is enqueued by calling fenced_upload(). Multiple uploads can be batched together, which will complete at the same time.

.. doxygenfunction:: vuk::Context::fenced_upload(std::span<BufferUpload> uploads)

.. doxygenfunction:: vuk::Context::fenced_upload(std::span<ImageUpload> uploads)

The result of the upload must be polled on to determine completion. This can be done once a frame, for example.

.. doxygenstruct:: vuk::Context::UploadResult
   :members:

Call :cpp:func:`vuk::Context::free_upload_resources()` to dispose of the temporary data:

.. doxygenfunction:: vuk::Context::free_upload_resources
   :outline: