.. _popxl-tensors:

Tensors
-------

.. autoclass:: popxl.tensor.Tensor
    :members:
    :undoc-members:
    :special-members: __init__

.. autoclass:: popxl.tensor.Constant
    :members:
    :undoc-members:
    :special-members: __init__

.. autoclass:: popxl.tensor.Variable
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

.. autodata:: popxl.tensor.HostScalarTensor
.. autodata:: popxl.tensor.HostTensor
.. autodata:: popxl.tensor.ScalarType
.. autodata:: popxl.tensor.TensorLike
