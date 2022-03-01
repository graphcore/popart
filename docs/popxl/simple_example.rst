.. _sec_simple_example:

Simple example
==============

To illustrate the PopXL concepts, here is a simple example.

You need to import the ``popxl`` module in order to use it:

.. code-block:: python

  import popxl


The IR in PopXL is represented by the class ``popxl.Ir``.
:numref:`simple_example` contains a basic example of how to construct such an object.

.. literalinclude:: ../user_guide/files/simple_addition_popxl.py
  :language: python
  :lines: 8-9,11-27
  :name: simple_example
  :caption: Simple example using PopXL

.. only:: html

    :download:`Download simple_addition_popxl.py <../user_guide/files/simple_addition_popxl.py>`

In PopXL an IR is essentially a collection of :py:class:`popxl.Graph` objects.
Each such graph contains a number of operations.
Each IR has a *main graph* that is constructed by default.
This main graph serves as the entry point for your model.
A main graph is obtained via :py:func:`popxl.main_graph` in the example above.

By adding operations within a ``with main`` context, the operations
are automatically added to the main graph.
In this example, three operations are added: ``host_load``, ``add`` and ``host_store``.

In this model we created two device-to-host streams ``input0`` and ``input1``
and one host-to-device stream ``output``.
The ``host_load`` operations are used to stream data from the host to
the device populating tensors ``a`` and ``b``, respectively.
Another operation, ``add``, then adds these two tensors together.
Finally, the ``host_store`` streams the result data back from the device to the host.
