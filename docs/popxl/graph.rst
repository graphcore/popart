.. _sec_graphs:

Graphs
======

.. _sec_maingraphs:

Main graph
-----------

You can create the main graph of an IR by calling ``popxl.main_graph``.
The returned main graph can be used as a context to include its operations
and tensors.

.. _sec_subgraphs:

Subgraphs
-----------

You can create a :ref:`subgraph <graph_concept>` in PopXL by calling, for example, ``ir.create_graph("graph")``.
You then connect it with the calling graph using ``ops.call``.
In PopXL, you have access to ``create_graph`` before you call a graph with
``ops.call``, which gives you the flexibility to manipulate the graph.

:numref:`code_basic_subgraph_popxl` shows a basic example about how to create and call subgraphs.
A subgraph is created and called instead of directly calling the Python function
``increment_fn``.

.. literalinclude:: ../user_guide/files/basic_subgraph_popxl.py
  :language: python
  :name: code_basic_subgraph_popxl
  :caption: Example to create and call subgraphs
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download basic_subgraph_popxl.py <../user_guide/files/basic_subgraph_popxl.py>`

Creating a subgraph
-------------------

You can create a subgraph by calling the function ``ir.create_graph``.
You can use the same function to create multiple subgraphs.
In the example in :numref:`code_create_multi_subgraphs_from_same_func_popxl`, two different graphs are created
for different input tensors, ``w1`` and ``w2``, which have different shapes.

.. literalinclude:: ../user_guide/files/create_multi_subgraphs_from_same_func_popxl.py
  :language: python
  :name: code_create_multi_subgraphs_from_same_func_popxl
  :caption: Example of creating multiple subgraphs with same function
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download create_multi_subgraphs_from_same_func_popxl.py
    <../user_guide/files/create_multi_subgraphs_from_same_func_popxl.py>`

You can also create the subgraph with an additional graph input with ``popxl.graph_input``
in its Python function. The ``graph_input`` creates a new input tensor for the
subgraph. An example can be found in :numref:`multi_call_graph_input_example`.

Calling a subgraph
----------------------

After you have created a subgraph, you can invoke it with an op ``call`` with
the provided input tensors as follows:

.. code-block:: python

  call(subgraph: Graph,
      *inputs: Union[Tensor, List[Tensor]],
      inputs_dict: Optional[Mapping[Tensor, Tensor]] = None
      ) -> Union[None, Tensor, Tuple[Tensor, ...]]:

``inputs`` are the inputs the subgraph requires and they must
be in the same order as in ``create_graph``. If you are not sure about the order
of the subgraph internal tensors that are defined by
``popxl.graph_input``, you can use ``inputs_dict`` to
provide the mapping between the subgraph tensors and the parent graph tensors.

.. :note:: Each subgraph can be called from multiple call sites, but it is compiled only once to avoid redundant code.

:numref:`multi_call_graph_input_example` shows an example of a graph being called multiple times with different variables.
In this example, the subgraph was created with an additional graph input ``value``.
When you call this subgraph, you will have to pass a tensor to the subgraph
for this input as well. You can use it to instantiate the weights of layers internally.

.. literalinclude:: ../user_guide/files/multi_call_graph_input_popxl.py
  :language: python
  :start-after: Op begin
  :end-before: Op end
  :name: multi_call_graph_input_example
  :caption: Example of a graph being called multiple times with different variables

.. only:: html

    :download:`Download multi_call_graph_input_popxl.py <../user_guide/files/multi_call_graph_input_popxl.py>`


You can call a graph and get the information about the call site by using the op
``ops.call_with_info`` instead of ``ops.call``. ``ops.call_with_info`` returns a
``CallInfo`` object that provides extra information about the call site. For
instance, you can get the graph being called using ``called_graph``.
``inputs`` and ``outputs`` return the input tensors and
output tensors respectively. You can also obtain the input and output tensors at
a given index using ``parent_input(index)`` and
``parent_output(index)`` respectively. You can find the input
subgraph tensor that corresponds to a parent tensor using
``parent_to_graph (parent_tensor)``.
``graph_to_parent(subgraph_tensor)`` provides an input or output tensor in
``called_graph`` that associates the input or output tensor in the parent graph.

With the ``CallInfo`` object, you can use ``set_parent_input_modified(subgraph_tensor)`` to specify
that the input ``subgraph_tensor`` can be modified by this ``call_with_info`` op. This provides
support for in-place variable updates like in :numref:`code_call_with_info_popxl`. After calling the subgraph, the value
of variable tensor ``x`` is changed to 2.

.. literalinclude:: ../user_guide/files/call_with_info_popxl.py
  :language: python
  :name: code_call_with_info_popxl
  :caption: Example of ``call_with_info`` op
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download call_with_info_popxl.py <../user_guide/files/call_with_info_popxl.py>`

The op ``call_with_info`` is helpful when building and optimizing the backward graph.
More details are given in :numref:`autodiff`.


Calling a subgraph in a loop
---------------------------------

You can use ``ops.repeat`` to create a loop, see :py:func:`popxl.ops.repeat()`.

.. code-block:: python

    repeat(graph: Graph,
           repeat_count: int,
           *inputs: Union[Tensor, Iterable[Tensor]],
           inputs_dict: Optional[Mapping[Tensor, Tensor]] = None
           ) -> Tuple[Tensor, ...]:

It calls a subgraph ``graph`` for ``repeat_count`` number of times.
Its inputs come from two arguments:

 - ``inputs`` that denotes the inputs passed to the subgraph function and,
 - ``inputs_dict`` that denotes a mapping from internal tensors in the subgraph being called to tensors at the call site in parent graph.

Both inputs from ``inputs`` and ``inputs_dict``
are loop-carried inputs. This means that they are copied into the subgraph as inputs
before the first iteration run. The outputs of each iteration are copied to the
inputs of the next iteration as shown in :numref:`fig_repeat_op`. The outputs of the last
iteration serve as the outputs of this ``repeat`` op.

.. figure:: images/repeat_op.png
  :name: fig_repeat_op
  :align: center
  :alt: repeat op of subgraph

  Repeat op subgraph

The ``repeat`` op requires the number of the subgraph inputs including the ``inputs`` and
the ``inputs_dict`` to be at least the number of outputs.

.. note:: This operation requires the repeat count to be greater than 0.

In :numref:`code_repeat_subgraph_popxl_0`, the subgraph ``increment_graph``
from ``increment_fn`` is called twice. The input ``x`` is incremented twice by
``value``. After the first iteration, the outputs ``x + value`` and ``value``
are copied to the inputs for the second iteration.

.. literalinclude:: ../user_guide/files/repeat_subgraph_popxl_0.py
  :language: python
  :name: code_repeat_subgraph_popxl_0
  :caption: Example of ``repeat`` op to increment a tensor by a fixed value
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download repeat_subgraph_popxl_0.py <../user_guide/files/repeat_subgraph_popxl_0.py>`


:numref:`code_repeat_subgraph_popxl_1` shows how to use the
``inputs_dict``. The callable class ``Linear`` defines a linear
layer. The subgraph ``linear_graph`` is created from the module ``build``
method.

.. literalinclude:: ../user_guide/files/repeat_subgraph_popxl_1.py
  :language: python
  :name: code_repeat_subgraph_popxl_1
  :caption: Example of ``repeat`` op using ``inputs_dict``
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download repeat_subgraph_popxl_1.py <../user_guide/files/repeat_subgraph_popxl_1.py>`
