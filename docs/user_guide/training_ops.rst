Training operations
-------------------

To train a network, a list of Loss operations must be passed to the `Session`.
These typically take named input tensors and name their outputs.

Losses
~~~~~~

.. doxygenclass:: L1Loss

.. doxygenclass:: NllLoss


Optimizers
~~~~~~~~~~

.. doxygenclass:: ConstSGD

.. doxygenclass:: SGD

