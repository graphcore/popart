Operations in a graph are connected by input and output tensors.
Each operation is applied to the input tensors, and optionally produces output tensors.
The supported operations are listed in :ref:`Available ops <available_ops>`.

You can add operations to a graph by calling operation methods within
the graph context as shown in the :ref:`Simple example<simple_example>`.
Within the context of the ``main`` graph, ``host_load``, ``add`` and ``host_store``
are added to the main graph.

Data input and output
"""""""""""""""""""""

In ``popart.ir``, you can create data transfer stream from host to device by using

.. code-block:: python

  h2d_stream(shape: Iterable[int], dtype: dtype, name: Optional[str] = None)

then load data through the host to device stream by using

.. code-block:: python

  host_load(h2d_stream: HostToDeviceStream, name: Optional[str] = None)

where the ``h2d_stream`` handles the stream, and the ``name`` is the
name of the returned tensor.

Similarly, you can create data transfer stream from device to host by using

.. code-block:: python

  d2h_stream(shape: Iterable[int], dtype: dtype, name: Optional[str] = None)

then store data from device to host by using

.. code-block:: python

  host_store(d2h_stream: DeviceToHostStream, t: Tensor)

where the ``t`` is the tensor to copy to host.
