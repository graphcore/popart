Training operations
-------------------

To train a network, a scalar loss TensorId must be passed to the `Session`.
This can be the output of any operator in the model.

Optimisers
~~~~~~~~~~

.. doxygenclass:: popart::ConstSGD
  :members:

.. doxygenclass:: popart::SGD
  :members:
