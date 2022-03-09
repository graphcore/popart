.. _sec_context:

Context manager
===============

..
  Using the context manager
  ~~~~~~~~~~~~~~~~~~~~~~~~~

  - Explain how to use our context manager, and why/when you want to use it.

You can get your the graph context by using:

* ``get_current_graph()`` also called ``gcg()``.

  Returns the current graph from the current context.

* ``get_main_graph()``, also called ``gmg()``.

  Returns the current main graph of the current context.

You can also use context managers listed here in ``popxl``:

* ``ipu(ipu: int)``.

  Sets the ipu on ops created in this context. Internally it uses poplar
  `virtual graph <https://docs.graphcore.ai/projects/graphcore-glossary/en/latest/index.html#term-Virtual-graph>`_
  id as ipu id. This indicates which IPU device the operations within the context should be mapped to.
  For instance, you can use ``with ipu(0)`` to include the operations you want to put on the first IPU requested.

  The usage of virtual graph in PopXL is similar to the the `virtual graph in Poplar <https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/poplar_programs.html#virtual-graphs>`_,
  except that PopXL requires to have only one virtual compute graph per IPU.

* ``in_sequence()``.

  Forces operations created in its context to be executed in the same order as they are created.
  In the example below, the ``in_sequence()`` guarantees the ``host_restore`` is executed after
  the ``copy_var_update``. The ``copy_var_update_`` updates the value of ``x`` inplace by ``b``.
  So the value of ``x`` is 5. If the ``in_sequence()`` is not used, this will cause inplacing
  ambiguity between the ``copy_var_update`` and ``host_store``.

.. literalinclude:: ../user_guide/files/in_sequence_popxl.py
  :language: python
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download in_sequence_popxl.py <../user_guide/files/in_sequence_popxl.py>`

* ``name_scope(name)``.

  Sets the name scope for the operations created within its scope.

You can also use contexts ``merge_exchange`` and ``io_tile_exchange`` to do
transforms on graphs. See transform section for more details.
