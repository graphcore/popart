.. _sec_operations:

Operations
==========

Operations in a graph are connected by input and output tensors.
Each operation is applied to the input tensors, and optionally produces output tensors.
The supported operations are listed in :numref:`popart_ir_ops_available_ops`.

You can add operations to a graph by calling operation methods within
the graph context as shown in the :numref:`simple_example`.
Within the context of the ``main`` graph, ``host_load``, ``add`` and ``host_store``
are added to the main graph.

Data input and output
---------------------

In ``popart.ir``, you can create a data transfer stream from the host to the device by using:

.. code-block:: python

  h2d_stream(shape: Iterable[int], dtype: dtype, name: Optional[str] = None)

Then load data through the host to the device stream by using:

.. code-block:: python

  host_load(h2d_stream: HostToDeviceStream, name: Optional[str] = None)

Where the ``h2d_stream`` handles the stream, and the ``name`` is the
name of the returned tensor.

Similarly, you can create a data transfer stream from the device to the host by using:

.. code-block:: python

  d2h_stream(shape: Iterable[int], dtype: dtype, name: Optional[str] = None)

Then store data from device to host by using:

.. code-block:: python

  host_store(d2h_stream: DeviceToHostStream, t: Tensor)

Where the ``t`` is the tensor to be copied to the host.


List of available operations
----------------------------

The operations currently supported in ``popart.ir`` are listed in :numref:`popart_ir_ops_available_ops`, :numref:`popart_ir_ops_collectives_available_ops` and :numref:`popart_ir_ops_var_updates_available_ops`.

.. include:: ../popartir_supported_ops_gen.rst
