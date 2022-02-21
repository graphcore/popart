.. _sec_simple_example:

Simple example
--------------

You need to import the ``popart.ir`` module in order to use it:

.. code-block:: python

  import popart.ir as pir


The IR in ``popart.ir`` is represented by the class ``popart.ir.Ir``.
:numref:`simple_example` contains a basic example of how to construct such an object.

.. literalinclude:: ../user_guide/files/simple_addition_popart_ir.py
  :language: python
  :lines: 8-9,11-27
  :name: simple_example
  :caption: Simple example using ``popart.ir``

.. only:: html

    :download:`Download simple_addition_popart_ir.py <../user_guide/files/simple_addition_popart_ir.py>`

In ``popart.ir`` an IR is essentially a collection of :py:class:`popart.ir.Graph` objects.
Each such graph contains a number of operations.
Each IR has a *main graph* that is constructed by default.
This main graph serves as the entry point for your model.
A main graph is obtained via :py:func:`popart.ir.main_graph` in the example above.

By adding operations within a ``with main`` context, the operations
are automatically added to the main graph.
In this example, three operations are added: ``host_load``, ``add`` and ``host_store``.

In this model we created two device-to-host streams ``input0`` and ``input1``
and one host-to-device stream ``output``.
The ``host_load`` operations are used to stream data from the host to
the device populating tensors ``a`` and ``b``, respectively.
Another operation, ``add``, then adds these two tensors together.
Finally, the ``host_store`` streams the result data back from the device to the host.

.. _sec_data_types:

Data types
----------

Currently, ``popart.ir`` supports the data types listed in :numref:`ir_datatypes_table`.
These data types are defined in ``popart.ir`` directly and
will be converted to their IPU-compatible data type. Note that the ``int64``
and ``uint64`` will be downcast to ``int32`` and ``uint32`` respectively
if the session option ``enableSupportedDataTypeCasting`` is set to ``True``.

.. include:: popartir_datatype.rst

.. _sec_tensors:

Tensors
-------

.. include:: popartir_tensor.rst


.. _sec_operations:

Operations
----------

.. include:: popartir_op.rst

..
  Adding if operations
  --------------------

  - Explain how you add an IfOp in `popart.ir`.

..
  Using the context manager
  -------------------------

  - Explain how to use our context manager, and why/when you want to use it.

.. _sec_graphs:

Graphs
------

.. _sec_maingraphs:

Main graphs
...........

This section describes how to create the main graph.


.. _sec_subgraphs:

Subgraphs
.........

.. include:: popartir_subgraph.rst


.. include:: popartir_transforms.rst

..
  Running a model created in `popart.ir`
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - Explain how you can run a popart.ir IR.
  - Describe how to set up DataFlow, Sessions, anchors, any assumptions etc.
  - Give an example
  - Explain any constraints.
