Simple example
^^^^^^^^^^^^^^

To use ``popart.ir`` you first need to import it as a Python package:

.. code-block:: python

  import popart.ir as pir

Note that we typically import ``popart.ir`` as ``pir`` for brevity.

As explained previously, the main purpose of this package is creating and manipulating IRs,
which are represented by the class ``pir.Ir``.
See below for a basic example of how to construct such an object.

.. literalinclude:: files/simple_addition_popart_ir.py
  :language: python
  :lines: 3-4,6-22

.. only:: html

    :download:`files/simple_addition_popart_ir.py`

In ``popart.ir`` an IR is essentially a collection of ``pir.Graph`` objects.
Each such graph contains a number of operations.
Each IR has a *main graph* that is constructed by default.
This main graph serves as the entry point for your model.
A main graph is obtained via ``ir.main_graph()`` in the example above.

By adding operations within a ``with main`` context, the operations
are automatically added to the main graph.
In this example, three operations added: ``host_load``, ``add`` and ``host_store``.

In this model we created two device-to-host streams ``input0`` and ``input1``
and one host-to-device stream ``output``.
The ``host_load`` operations are used to stream data from the host to
the device populating tensors ``a`` and ``b``, respectively.
Another operation, ``add``, then sums these two tensors.
Finally, the ``host_store`` streams the result data back from the device to the host.


Data types
^^^^^^^^^^

Currently, ``popart.ir`` supports the data types listed in :numref:`ir_datatypes_table`.
These data types are defined in ``popart.ir`` directly and
will be converted to their IPU-compatible data type. Note that the ``int64``
and ``uint64`` will be downcast to ``int32`` and ``uint32`` respectively
if the session option ``enableSupportedDataTypeCasting`` is set to ``True``.

.. include:: popartir_datatype.rst

Tensors
^^^^^^^

.. include:: popartir_tensor.rst


..
  Adding operations to a graph
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - Explain how you can add ops to a graph (we haven't explained subgraphs at this point, so add to the main graph).
  - Explain what in_sequence does and show how to use it.

..
  Data input and output
  ^^^^^^^^^^^^^^^^^^^^^

  - Explain what host_load and host_store do and how to add them.

..
  Available operations
  ^^^^^^^^^^^^^^^^^^^^

  - Detail supported operations (with links to the Python API).

..
  Creating subgraphs
  ^^^^^^^^^^^^^^^^^^

  - Explain how you create a subgraph.

..
  Calling subgraphs
  ^^^^^^^^^^^^^^^^^

  - Explain how you add a CallOp in `popart.ir`.

..
  Adding loops operations
  ^^^^^^^^^^^^^^^^^^^^^^^
  - Explain how you add a LoopOp in `popart.ir`.

..
  Adding if operations
  ^^^^^^^^^^^^^^^^^^^^

  - Explain how you add an IfOp in `popart.ir`.

..
  Using the context manager
  ^^^^^^^^^^^^^^^^^^^^^^^^^

  - Explain how to use our context manager, and why/when you want to use it.

..
  Applying transforms
  ^^^^^^^^^^^^^^^^^^^

  - Explain what transforms are available and how you use them.

..
  Autodiff
  ^^^^^^^^

  - Specialised section on autodiff.

..
  Running a model created in `popart.ir`
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - Explain how you can run a popart.ir IR.
  - Describe how to set up DataFlow, Sessions, anchors, any assumptions etc.
  - Give an example
  - Explain any constraints.
