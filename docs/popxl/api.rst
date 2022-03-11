.. _popxl-api:

Python API Reference
====================

.. warning::
     This Python module is currently experimental and may be subject to change
     in future releases in ways that are not backwards compatible and without
     deprecation warnings.

The ``popxl`` module provides access to the intermediate representation (IR)
for a computational graph of a model. The PopART IR is the intermediate
representation of models created with PopART.

This is an experimental Python module that allows you to create (and to a
limited degree manipulate) PopART IRs directly.


.. autoclass:: popxl.Ir
    :members:
    :undoc-members:
    :special-members:

.. _popxl-contexts:

Contexts
--------

.. autoclass:: popxl.Context
    :members:
    :undoc-members:
    :special-members:

.. autofunction:: popxl.gcg
.. autofunction:: popxl.get_current_graph
.. autofunction:: popxl.get_main_graph
.. autofunction:: popxl.gmg
.. autofunction:: popxl.in_sequence
.. autofunction:: popxl.io_tiles
.. autofunction:: popxl.ipu
.. autofunction:: popxl.name_scope
.. autofunction:: popxl.op_debug_context

.. _popxl-dtypes:

Data types
----------

.. autoclass:: popxl.dtype
    :members:
    :undoc-members:
    :special-members:

.. _popxl-graphs:

Graphs
-------

.. autoclass:: popxl.Graph
    :members:
    :undoc-members:
    :special-members:

.. _popxl-modules:

Modules
-------

.. autoclass:: popxl.Module
    :members:
    :undoc-members:
    :special-members:

.. _popxl-random-seeds:

Random seeds
-------------

.. autofunction:: popxl.create_seeds
.. autofunction:: popxl.two_uint32_to_uint64
.. autofunction:: popxl.uint64_to_two_uint32

.. _popxl-remote-buffers:

Remote buffers
--------------

.. autoclass:: popxl.RemoteBuffer
    :members:
    :undoc-members:
    :special-members:

.. autofunction:: popxl.remote_buffer

.. _popxl-streams:

Streams
-------

.. autoclass:: popxl.DeviceToHostStream
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popxl.HostToDeviceStream
    :members:
    :undoc-members:
    :special-members:

.. _popxl-tensors:

Tensors
-------

.. autoclass:: popxl.Tensor
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popxl.Constant
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popxl.Variable
    :members:
    :undoc-members:
    :special-members:

.. autofunction:: popxl.constant
.. autofunction:: popxl.remote_replica_sharded_variable
.. autofunction:: popxl.remote_variable
.. autofunction:: popxl.replica_sharded_variable
.. autofunction:: popxl.graph_input
.. autofunction:: popxl.graph_output
.. autofunction:: popxl.variable

.. autodata:: popxl.HostScalarTensor
.. autodata:: popxl.HostTensor
.. autodata:: popxl.ScalarType
.. autodata:: popxl.TensorLike

.. _popxl-tensor-locations:

Tensor Locations
----------------

.. autoclass:: popxl.ExecutionContext
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popxl.ReplicatedTensorSharding
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popxl.TensorLocation
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popxl.TensorStorage
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popxl.TileSet
    :members:
    :undoc-members:
    :special-members:

.. _popxl-transforms:

Transforms
----------

.. autoclass:: popxl.ExpectedConnectionType
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popxl.GradGraphInfo
    :members:
    :undoc-members:
    :special-members:

.. autofunction:: popxl.autodiff
.. autofunction:: popxl.io_tile_exchange
.. autofunction:: popxl.merge_exchange


.. _available_ops:

Ops available in PopXL
----------------------

.. automodule:: popxl.ops
    :members:
    :undoc-members:
    :special-members:
