.. _sec_supported_operations:

Supported operations
====================

Operations (or ops) in a graph are connected by input and output tensors. Each
operation is applied to the input tensors, and optionally produces output
tensors. The supported operations are listed in
:numref:`sec_popxl_ops_available_ops`.

You can add operations to a graph by calling operation methods within the graph
context as shown in :numref:`simple_example`. In this example, within the
context of the ``main`` graph, ``host_load``, ``add`` and ``host_store`` are
added to the main graph.

Data input and output
---------------------

In PopXL, you can create a data transfer stream from the host to the device with:

.. code-block:: python

  h2d_stream(shape: Iterable[int], dtype: dtype, name: Optional[str] = None)

Then, load data through the host to the device stream with:

.. code-block:: python

  host_load(h2d_stream: HostToDeviceStream, name: Optional[str] = None)

where ``h2d_stream`` handles the stream, and ``name`` is the
name of the returned tensor.

Similarly, you can create a data transfer stream from the device to the host:

.. code-block:: python

  d2h_stream(shape: Iterable[int], dtype: dtype, name: Optional[str] = None)

Then, store data from device to host with:

.. code-block:: python

  host_store(d2h_stream: DeviceToHostStream, t: Tensor)

where ``t`` is the tensor to be copied to the host. Note that you require a
separate ``host_load`` or ``host_store`` for each tensor and transfer to or from
the device respectively. However, the transfers will be internally merged into one for efficiency if the op's schedule allows it.

.. _sec_popxl_ops_available_ops:

List of available operations
----------------------------

The operations currently supported in PopXL are listed in :numref:`popxl_ops_available_ops`, :numref:`popxl_ops_collectives_available_ops` and :numref:`popxl_ops_var_updates_available_ops`.

.. include:: ../popxl_supported_ops_gen.rst
