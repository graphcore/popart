Training operations
-------------------

To train a network, a list of Loss operations must be passed to the `Session`.
These typically take named input tensors and name their outputs, and specify the reduction to apply to the output.

Losses
~~~~~~

.. doxygenclass:: poponnx::L1Loss

.. doxygenclass:: poponnx::NllLoss


Optimizers
~~~~~~~~~~

.. doxygenclass:: poponnx::ConstSGD

.. doxygenclass:: poponnx::SGD

