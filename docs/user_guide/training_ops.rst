Training operations
-------------------

To train a network, a list of Loss operations must be passed to the `Session`.
These typically take named input tensors and name their outputs, and specify the reduction to apply to the output.

Losses
~~~~~~

.. doxygenclass:: popart::L1Loss

.. doxygenclass:: popart::NllLoss


Optimizers
~~~~~~~~~~

.. doxygenclass:: popart::ConstSGD

.. doxygenclass:: popart::SGD
