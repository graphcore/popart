.. _popart-ir-api:

PopART IR Python API (experimental)
===================================

.. warning::
     This Python module is currently experimental and may be subject to change
     in future releases in ways that are not backwards compatible and without
     deprecation warnings.


The ``popart.ir`` module provides access to the intermediate representation (IR)
for a computational graph of a model. The PopART IR is the intermediate representation
of models created with PopART.

This is an experimental PopART Python module that allows you to create (and to a limited degree manipulate) PopART IRs
directly.
This provides greater flexibility than is possible using the standard PopART API.


.. autoclass:: popart.ir.Ir
    :members:
    :undoc-members:
    :special-members:

.. _popart-ir-contexts:

Contexts
--------

.. autoclass:: popart.ir.Context
    :members:
    :undoc-members:
    :special-members:

.. autofunction:: popart.ir.gcg
.. autofunction:: popart.ir.get_current_graph
.. autofunction:: popart.ir.get_main_graph
.. autofunction:: popart.ir.gmg
.. autofunction:: popart.ir.in_sequence
.. autofunction:: popart.ir.io_tiles
.. autofunction:: popart.ir.ipu
.. autofunction:: popart.ir.name_scope
.. autofunction:: popart.ir.op_debug_context
.. autofunction:: popart.ir.pipeline_stage

.. _popart-ir-dtypes:

Data types
----------

.. autoclass:: popart.ir.dtypes
    :members:
    :undoc-members:
    :special-members:

.. _popart-ir-graphs:

Graphs
-------

.. autoclass:: popart.ir.Graph
    :members:
    :undoc-members:
    :special-members:

.. _popart-ir-modules:

Modules
-------

.. autoclass:: popart.ir.Module
    :members:
    :undoc-members:
    :special-members:

.. _popart-ir-random-seeds:

Random seeds
-------------

.. autofunction:: popart.ir.create_seeds
.. autofunction:: popart.ir.two_uint32_to_uint64
.. autofunction:: popart.ir.uint64_to_two_uint32

.. _popart-ir-remote-buffers:

Remote buffers
--------------

.. autoclass:: popart.ir.RemoteBuffer
    :members:
    :undoc-members:
    :special-members:

.. autofunction:: popart.ir.remote_buffer

.. _popart-ir-streams:

Streams
-------

.. autoclass:: popart.ir.DeviceToHostStream
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popart.ir.HostToDeviceStream
    :members:
    :undoc-members:
    :special-members:

.. _popart-ir-tensors:

Tensors
-------

.. autoclass:: popart.ir.Tensor
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popart.ir.Constant
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popart.ir.Variable
    :members:
    :undoc-members:
    :special-members:

.. autofunction:: popart.ir.constant
.. autofunction:: popart.ir.remote_replica_sharded_variable
.. autofunction:: popart.ir.remote_variable
.. autofunction:: popart.ir.replica_sharded_variable
.. autofunction:: popart.ir.subgraph_input
.. autofunction:: popart.ir.subgraph_output
.. autofunction:: popart.ir.variable

.. autodata:: popart.ir.HostScalarTensor
.. autodata:: popart.ir.HostTensor
.. autodata:: popart.ir.ScalarType
.. autodata:: popart.ir.TensorLike

.. _popart-ir-tensor-locations:

Tensor Locations
----------------

.. autoclass:: popart.ir.ExecutionContext
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popart.ir.ReplicatedTensorSharding
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popart.ir.TensorLocation
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popart.ir.TensorStorage
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popart.ir.TileSet
    :members:
    :undoc-members:
    :special-members:

.. _popart-ir-transforms:

Transforms
----------

.. autoclass:: popart.ir.ExpectedConnectionType
    :members:
    :undoc-members:
    :special-members:

.. autoclass:: popart.ir.GradGraphInfo
    :members:
    :undoc-members:
    :special-members:

.. autofunction:: popart.ir.autodiff
.. autofunction:: popart.ir.io_tile_exchange
.. autofunction:: popart.ir.merge_exchange


.. _available_ops:

Ops available in popart.ir
--------------------------

.. automodule:: popart.ir.ops
    :members:
    :undoc-members:
    :special-members:
