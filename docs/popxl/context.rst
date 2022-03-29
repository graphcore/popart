.. _sec_context:

Context manager
===============

PopXL contains a number of context managers.

..
  Using the context manager
  ~~~~~~~~~~~~~~~~~~~~~~~~~

  - Explain how to use our context manager, and why/when you want to use it.

You can get the graph context with:

* :py:func:`popxl.get_current_graph` (or :py:func:`popxl.gcg`): returns the current graph from the current context.

* :py:func:`popxl.get_main_graph` (or :py:func:`popxl.gmg`): returns the main graph of the current context.

You can also use the following context managers in ``popxl``:

* :py:func:`popxl.ipu`: Sets the IPU id on ops created in this context. Internally, it
  uses the Poplar `virtual graph
  <https://docs.graphcore.ai/projects/graphcore-glossary/en/latest/index.html#term-Virtual-graph>`_
  id as the IPU id. This indicates which IPU the operations within the
  context should be mapped to. For instance, you can use ``with ipu(0)`` to
  include the operations you want to put on the first IPU requested.

  The usage of virtual graph in PopXL is similar to the `virtual graph in Poplar <https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/poplar_programs.html#virtual-graphs>`_,
  except that PopXL requires only one virtual computational graph per IPU.

* :py:func:`popxl.in_sequence`: Forces operations created in its context to be executed in
  the same order as they are created. In :numref:`code-in-sequence-example`,
  ``in_sequence()`` guarantees ``host_restore`` is executed after the
  ``copy_var_update``. ``copy_var_update_`` updates the value of ``x`` in-place
  with ``b``. So the value of ``x`` is 5. If ``in_sequence()`` is not used,
  this will cause ambiguity with the in-place operation between
  ``copy_var_update`` and ``host_store``.

.. literalinclude:: files/in_sequence.py
  :language: python
  :start-after: Op begin
  :end-before: Op end
  :name: code-in-sequence-example
  :caption: Example of using ``in_sequence`` context manager

.. only:: html

    :download:`Download in_sequence.py <files/in_sequence.py>`

* :py:func:`popxl.name_scope`: sets the name scope for the operations created within this context.

* :py:func:`popxl.io_tiles`: executes for the operations created within this context on the I/O tiles of the current IPU.


You can also use the contexts :py:func:`popxl.merge_exchange` and :py:func:`popxl.io_tile_exchange` to do transforms on graphs. :numref:`sec_transforms` contains details.
