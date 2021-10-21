.. warning::
     The `popart.ir` Python module is currently experimental and may be subject to change
     in future releases in ways that are backwards incompatible without
     deprecation warnings.

.. warning::
     Due to the experimental nature of `popart.ir` the documentation provided in
     this section is incomplete.

As an alternative to using the ONNX builder to create models, ``popart.ir`` is
an experimental PopART python module through which it is possible to create
(and to a limited degree manipulate) PopART IRs directly.

..
  NOTE: Comments in .rst are '..' followed by a new line and an indentation.
  As you write content for a section heading that is commented out, please
  un-comment the heading also.

..
  Creating a model with `popart.ir`
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - Explain what popart.ir.Ir is and that it has a main graph.
  - Add a very basic example that creates an IR and main graph.

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
