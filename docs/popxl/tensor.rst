.. _sec_tensors:

Tensors
=======


The concepts to tensors were introduced in :numref:`sec_concept_tensors`.
You define a tensor with shape, data type and optional initialisation data.
A tensor has zero or more consumer operations and up to one producer operation.

There are three types of tensors in PopXL:

  - Constant
  - Variable
  - Intermediate

An intermediate tensor is the output of an operation.
Variable tensors and constant tensors are initialised with data.
For instance, in the example :numref:`tensor_addition_code`, ``a`` is a variable tensor,
``b`` is a constant tensor, and ``o`` is an intermediate tensor.


.. literalinclude:: ../user_guide/files/tensor_addition_popxl.py
  :language: python
  :name: tensor_addition_code
  :caption: Example of tensor addition
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download tensor_addition_popxl.py <../user_guide/files/tensor_addition_popxl.py>`

.. _sec_tensors_constant:

Constant tensors
----------------

A constant tensor is initialised with data during graph creation with
:py:func:`popxl.constant()`. This tensor cannot change during the runtime of a
model. You can also use Python numeric literals in PopXL. These literals are
implicitly converted to constant tensors. For example:

.. code-block:: python

  b = popxl.constant(1, dtype=popxl.int8, name="constant_b")
  o = a + b

can also be written as:

.. code-block:: python

  o = a + 1

.. _sec_tensors_variable:

Variable tensors
----------------

Variable tensors are always live in IPU memory and this memory does not get freed during execution. Therefore, a variable tensor is used to represents trainable parameters in a model or non-trainable optimizer states.

You create and initialize variable tensors in the scope of the
main graph. You can add a variable tensor to the main graph using
:py:func:`popxl.variable()`.

To enable flexible interaction, you can read or write variable tensors on
the IPU at runtime using :py:func:`popart_core.Session.readWeights()` and
:py:func:`popart_core.Session.writeWeights()` methods respectively.

Note that, you have to copy the initial value of a variable tensor to the IPU
from the host before running the graph with :py:func:`popart_core.Session.weightsFromHost()`.

.. _sec_tensors_intermediate:

Intermediate tensors
--------------------

An intermediate tensor is produced by an operation, which means it is not
initialised with data. It stays live in IPU memory from the point at which it is
produced until the last time it is consumed.
