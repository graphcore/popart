You define a tensor with shape, data type and optional initialisation data.
A tensor has zero or more consumer operations and up to one producer operation.

There are three types of tensors in ``popart.ir``:

  - Constant
  - Variable
  - Intermediate

An intermediate tensor is the output of an operation.
Variables and constants are initialised with data.
For instance, in the example :numref:`tensor_addition_code`, ``a`` is a variable tensor,
``b`` is a constant tensor, and ``o`` is an intermediate tensor.


.. literalinclude:: files/tensor_addition_popart_ir.py
  :language: python
  :name: tensor_addition_code
  :caption: Example of tensor addition
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`files/tensor_addition_popart_ir.py`

Constant
........

A constant tensor is initialised with data during graph creation.
This tensor cannot change during the runtime of a model.
You can also use Python numeric literals in ``popart.ir``.
These literals are implicitly converted to constant tensors.
That is, this:

.. code-block:: python

  b = pir.constant(1, dtype=pir.int8, name="constant_b")
  o = a + b

can also be written as:

.. code-block:: python

  o = a + 1

Variable
........

A variable tensor represents trainable parameters in a model
or non-trainable optimizer states.
You create and initialize variable tensors in the main graph scope.
You can add a variable to the main graph using ``pir.variable``.

To enable flexible interaction, you can read or write variables on
the IPU at runtime using the ``session.readWeights()`` and
``session.writeWeights()`` methods respectively.
Therefore, variables are always live in IPU memory and don't get freed
during execution.

Note that, you have to copy the initial value of a variable to IPU device
from the host before running the graph by using ``session.weightsFromHost()``.


Intermediate
............

An intermediate tensor is produced by an operation, which means it is not initialised
with data. It stays live in IPU memory from when it is produced until the last time
it is consumed.
