.. _popxl-tensors:

Tensors
-------

.. autoclass:: popxl.Tensor
    :members:
    :undoc-members:
    :special-members: __init__

.. autoclass:: popxl.Constant
    :members:
    :undoc-members:
    :special-members: __init__

.. autoclass:: popxl.Variable
    :members:
    :undoc-members:
    :special-members: __init__

.. autofunction:: popxl.constant
.. autofunction:: popxl.variable
.. autofunction:: popxl.remote_variable
.. autofunction:: popxl.replica_sharded_buffer
.. autofunction:: popxl.remote_replica_sharded_variable
.. autofunction:: popxl.replica_sharded_variable
.. autofunction:: popxl.graph_input
.. autofunction:: popxl.graph_output

.. autodata:: popxl.HostScalarTensor
.. autodata:: popxl.HostTensor
.. autodata:: popxl.ScalarType
.. autodata:: popxl.TensorLike
