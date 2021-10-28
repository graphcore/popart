.. warning::
     The ``popart.ir`` Python module is currently experimental and may be subject to change
     in future releases in ways that are backwards incompatible without
     deprecation warnings.

.. warning::
     Due to the experimental nature of ``popart.ir`` the documentation provided in
     this section is incomplete.

..
  NOTE: Comments in .rst are '..' followed by a new line and an indentation.
  As you write content for a section heading that is commented out, please
  un-comment the heading also.

As an alternative to using the ONNX builder to create models, ``popart.ir`` is
an experimental PopART Python module which you can use to create
(and, to a limited degree, manipulate) PopART models directly.

PopART models are represented using an intermediate representation (IR).
The ``popart.ir`` package allows you to manipulate these IRs.

Creating a model with ``popart.ir``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

..
  Data types
  ^^^^^^^^^^

  - Add notes about popart.ir data types here.

..
  Variables
  ^^^^^^^^^

  - Explain what variables are and how to add them.

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
  =================

  - Explain how you add a CallOp in `popart.ir`.

..
  Adding loops operations
  =======================
  - Explain how you add a LoopOp in `popart.ir`.

..
  Adding if operations
  ====================

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
  ========

  - Specialised section on autodiff.

..
  Running a model created in `popart.ir`
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - Explain how you can run a popart.ir IR.
  - Describe how to set up DataFlow, Sessions, anchors, any assumptions etc.
  - Give an example
  - Explain any constraints.
