.. _ch_graphs:

Graphs
======

.. _sec_maingraphs:

Main graph
----------

You can create the main graph of an IR by calling :py:attr:`~popxl.Ir.main_graph`.
The returned main graph can be used as a context to include its operations
and tensors.

.. _sec_subgraphs:

Graphs
------

You can create a subgraph (:numref:`graph_concept`) in PopXL by calling, for
example, :py:func:`~popxl.Ir.create_graph`. You then connect the
subgraph with the calling graph with the :py:func:`~popxl.ops.call` op. In
PopXL, you have access to :py:func:`~popxl.Ir.create_graph` before
you call a graph with :py:func:`~popxl.ops.call`, which gives you the
flexibility to manipulate the graph.

:numref:`code_basic_graph_popxl` shows a basic example for how to create and
call subgraphs. In the example, a subgraph is created and called instead of
directly calling the Python function ``increment_fn()``.

.. literalinclude:: files/basic_graph.py
  :language: python
  :name: code_basic_graph_popxl
  :caption: Example to create and call graphs
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download basic_graph.py <files/basic_graph.py>`

Creating a graph
----------------

You can create a subgraph by calling the function :py:func:`~popxl.Ir.create_graph`.
You can use the same function to create multiple subgraphs.
In the example in :numref:`code_create_multi_graphs_from_same_func_popxl`, two different graphs are created
for different input tensors, ``w1`` and ``w2``, which have different shapes.

.. literalinclude:: files/create_multi_graphs_from_same_func.py
  :language: python
  :name: code_create_multi_graphs_from_same_func_popxl
  :caption: Example of creating multiple graphs with same function
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download create_multi_subgraphs_from_same_func.py <files/create_multi_graphs_from_same_func.py>`

You can also create the subgraph with an additional graph input with :py:func:`~popxl.graph_input`
in its Python function. :py:func:`~popxl.graph_input` creates a new input tensor for the
subgraph. An example can be found in :numref:`multi_call_graph_input_example`.

Calling a graph
---------------

After you have created a subgraph, you can invoke it with :py:func:`~popxl.ops.call`. The input tensors are as follows:

.. code-block:: python

  call(graph: Graph,
      *inputs: Union[Tensor, List[Tensor]],
      inputs_dict: Optional[Mapping[Tensor, Tensor]] = None
      ) -> Union[None, Tensor, Tuple[Tensor, ...]]:

``inputs`` are the inputs the subgraph requires and they must
be in the same order as in :py:func:`~popxl.Ir.create_graph`. If you are not sure about the order
of the subgraph internal tensors that are defined by
:py:func:`~popxl.graph_input`, you can use ``inputs_dict`` to
provide the mapping between the subgraph tensors and the parent graph tensors.

.. :note:: Each graph can be called from multiple call sites, but it is compiled only once to avoid redundant code.

:numref:`multi_call_graph_input_example` shows an example of a graph being called multiple times with different inputs.
In this example, the subgraph was created with an additional graph input ``value``.
When you call this subgraph, you will have to pass a tensor to the subgraph
for this input as well. You can use it to instantiate the weights of layers internally.

.. literalinclude:: files/multi_call_graph_input.py
  :language: python
  :start-after: Op begin
  :end-before: Op end
  :name: multi_call_graph_input_example
  :caption: Example of a graph being called multiple times with different inputs

.. only:: html

    :download:`Download multi_call_graph_input.py <files/multi_call_graph_input.py>`


Instead of calling a graph with :py:func:`~popxl.ops.call`, you can call it and get the information about the call site with the op
:py:func:`~popxl.ops.call_with_info`. This op returns a
:py:obj:`~popxl.ops.CallSiteInfo` object that provides extra information about the call site. For
instance, you can get the graph being called using ``called_graph``.
``inputs`` and ``outputs`` return the input tensors and
output tensors respectively. You can also obtain the input and output tensors at
a given index with ``parent_input(index)`` and
``parent_output(index)`` respectively. You can find the input
graph tensor that corresponds to a parent tensor using
``parent_to_graph (parent_tensor)``.
``graph_to_parent(graph_tensor)`` provides an input or output tensor in
``called_graph`` that associates the input or output tensor in the parent graph.

With the :py:obj:`~popxl.ops.CallSiteInfo` object, you can use ``set_parent_input_modified(subgraph_tensor)`` to specify
that the input tensor ``subgraph_tensor`` can be modified by this :py:func:`~popxl.ops.call_with_info` op. This provides
support for in-place variable updates as in :numref:`code_call_with_info_popxl`. After calling the subgraph, the value
of the variable tensor ``x`` is changed to 2.

.. literalinclude:: files/call_with_info.py
  :language: python
  :name: code_call_with_info_popxl
  :caption: Example of ``call_with_info`` op
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download call_with_info.py <files/call_with_info.py>`

The op :py:func:`~popxl.ops.call_with_info` is helpful when building and optimizing the backward graph. More details are given in :numref:`autodiff`.


Calling a graph in a loop
-------------------------

You can use the op :py:func:`~popxl.ops.repeat` to create a loop.

.. code-block:: python

    repeat(graph: Graph,
           repeat_count: int,
           *inputs: Union[Tensor, Iterable[Tensor]],
           inputs_dict: Optional[Mapping[Tensor, Tensor]] = None
           ) -> Tuple[Tensor, ...]:

This calls a subgraph ``graph`` for ``repeat_count`` number of times.
Its inputs are:

 - ``inputs`` denotes the inputs passed to the subgraph function and,
 - ``inputs_dict`` denotes a mapping from internal tensors in the subgraph being called to tensors at the call site in the parent graph.

Both inputs from ``inputs`` and ``inputs_dict`` are "loop-carried" inputs. This
means that they are copied into the subgraph as inputs before the first
iteration is run. The outputs of each iteration are copied to the inputs of the
next iteration as shown in :numref:`fig_repeat_op`. The outputs of the last
iteration serve as the outputs of the :py:func:`~popxl.ops.repeat` op.

.. figure:: images/repeat_op.png
  :name: fig_repeat_op
  :align: center
  :alt: repeat op of graph

  Repeat op graph

The :py:func:`~popxl.ops.repeat` op requires that the number of the subgraph inputs, including the ``inputs`` and the ``inputs_dict``, to be at least the number of outputs.

.. note:: This operation requires the repeat count to be greater than 0.

In :numref:`code_repeat_graph_popxl_0`, the graph ``increment_graph``
from ``increment_fn`` is called twice. The input ``x`` is incremented twice by
``value``. After the first iteration, the outputs ``x + value`` and ``value``
are copied to the inputs for the second iteration.

.. literalinclude:: files/repeat_graph_0.py
  :language: python
  :name: code_repeat_graph_popxl_0
  :caption: Example of ``repeat`` op to increment a tensor by a fixed value
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download repeat_graph_0.py <files/repeat_graph_0.py>`


:numref:`code_repeat_graph_popxl_1` shows how to use the
``inputs_dict``. The callable class ``Linear`` defines a linear
layer. The subgraph ``linear_graph`` is created from the PopXL ``build``
method.

.. literalinclude:: files/repeat_graph_1.py
  :language: python
  :name: code_repeat_graph_popxl_1
  :caption: Example of ``repeat`` op using ``inputs_dict``
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download repeat_graph_1.py <files/repeat_graph_1.py>`

Graph replication
-----------------

For improved performance, multiple IPUs can run in data parallel mode.
In data parallel mode multiple replicas of the graph are run on separate sets of IPUs.

Replicas can be grouped, see :numref:`sec_replication_types`. By default, there is only one group.
Replicas in a group are loaded with the same values.

Most operations can use replica grouping to reduce over only the grouped replica graphs,
allowing for all replicas in a group to benefit from each other's updates.

Graph replication cannot be used with IPU Model targets.

To set the replication factor (the number of replicated graphs), you
can set the ``ir.replication_factor``.


Code loading from Streaming Memory
----------------------------------

By default, tile memory is required for the tensors in the graph and for the
executable code for the compiled graph. To help alleviate this memory pressure,
as with tensors, you can store the executable code in Streaming Memory and load it, when required,
back into executable memory on the tiles.

Note not all the code will be offloaded and re-loaded.  For example, Poplar will
decide whether mutable vertex state or global exchange code will remain always
live. The code that is not offloaded will just stay in executable memory, and executing the
graph will always work without the requirement to explicitly load those parts of
code onto the IPU.

Minimal example
~~~~~~~~~~~~~~~

In PopXL, this code loading happens at the granularity of :py:class:`Graph` objects, because
it is each ``Graph`` that is compiled into one or more ``poplar::Function`` objects, which is then
compiled into executable IPU code.

A minimal example follows:

.. literalinclude:: files/code_loading.py
  :language: python
  :name: code_loading
  :caption: Minimal example of code loading in PopXL
  :start-after: BEGIN_BASIC
  :end-before: END_BASIC

.. only:: html

    :download:`Download code_loading.py <files/code_loading.py>`

From the example, you can see that no remote buffer for the ``Graph`` code is
explicitly created by the user. Instead, the :py:func:`ops.remote_code_load` for
that graph from Streaming Memory tells PopXL to create that remote buffer implicitly
for you. Multiple ``ops.remote_code_load`` calls from Streaming Memory on the same Graph
will reuse the same remote buffer. Note it is your responsibility to remember to insert
the ``remote_code_load`` op, otherwise it will seem to PopXL that the user intends
to have the code always-live in executable memory as normal. The
:py:func:`in_sequence` context around the ``remote_code_load`` and ``call`` is
also mandatory to ensure the copy is scheduled before the call. The need for
this is explained later. The other possible values of the parameter
``destination`` are also explained later.

In the above example, all the ops and tensors will be lowered into Poplar as
usual, then the Poplar liveness analyser will try to minimise liveness across the
computation by reusing memory where available. In this case, the liveness
analyser will see that the code does not need to be live until the
``remote_code_load`` call. Therefore the code is "dead" from ``(1)`` until
``(2)``, and hence less memory is consumed during this time. After
the ``remote_code_load``, the code is considered live. We can call the ``Graph``
(that is, execute the code) as many times as we want --- the code is still on device.
At ``(3)``, we call the
``Graph`` for the final time. The Poplar liveness analyser *may* use this fact
to consider the code dead after this point, and again recycle that memory for
another use.

To summarise, the code is only live from ``(2)`` to ``(3)``, whereas without
code loading, the code would have been always-live.

Note that when we say the code is "dead" or "not live", it is not *guaranteed*
that the memory will indeed be reused for something else, only that it could be.
Any part of the compilation stack may choose to optimise the graph in a
different way instead if it believes doing so to be more beneficial.

Lastly, the fact that the ``remote_code_load`` and ``call`` are inside an
``in_sequence`` context is very important. Recall that, in PopXL, you are
building a data-flow graph of ops and tensors, and by default they will execute
in whatever order the internal scheduler decides best (it aims to minimise
liveness). Observe that there is no data-flow dependence between the
``remote_code_load`` and the ``call``, meaning there is no tensor that the
``remote_code_load`` produces that the ``call`` consumes. This means, without
the ``in_sequence``, they *could* be scheduled in any order, and if the ``call``
comes first, the Poplar liveness analyser will think the code needs to be
always-live (in the case of the above example). Therefore, failing to use
``in_sequence`` results in undefined behaviour with respect to the code
liveness, and the onus is on you to remember to use it.

..   PopXL does not yet support code loading on IO tiles. That is, you cannot
..   ``remote_code_load`` between IO and compute tiles, or from Streaming Memory to IO
..   tiles. Attempting to create such a ``remote_code_load`` op will throw an
..   error. This also prevents overlapping the ``remote_code_load`` IO with other
..   compute.

Controlling liveness between multiple calls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every time you call a graph, it signifies that the code should be in tile
executable memory since either the last
``remote_code_load(destination='executable')``, or if there is no previous
``remote_code_load(destination='executable')``, the start of the program, in other words
the code is always-live.

Every time you use ``remote_code_load`` to load a ``Graph`` into a location, it signifies
that the code did not need to be live in that location since the last call, or
if there is no previous ``call``, from the start of the program.

Together, this gives full control of the code liveness of your graphs. Say
you have repeated calls to a ``Graph`` and you want the code to always be dead
in between calls until the latest possible moment. You simply insert
``remote_code_load`` ops just before every ``call``. The following example
demonstrates this:


.. literalinclude:: files/code_loading.py
  :language: python
  :name: code_loading
  :caption: Code loading example with multiple loads and calls
  :start-after: BEGIN_COMPLEX
  :end-before: END_COMPLEX

.. only:: html

    :download:`Download code_loading.py <files/code_loading.py>`

Note in the example we do not copy back the code from device to Streaming Memory.
This is for two reasons. Firstly, the code has no mutable state, so it is valid
to just keep loading repeatedly from the same remote buffer. In Poplar, it is
possible for code to have "mutable vertex state", but currently Poplar will
never offload that part of the code anyway and keep it always-live. Secondly,
Poplar attempts no liveness analysis in Streaming Memory to reuse a buffer for
something else when it is not needed. If this were the case, copying the code to
device would effectively free that space in Streaming Memory; so since that space
*cannot* be reused, it is pointless to perform the copy. Therefore, there is
no API for copying code *to* Streaming Memory.

..
  Tile memory code buffers
  ~~~~~~~~~~~~~~~~~~~~~~~~

  .. _Tile memory code buffers:

  As well as code buffers in Streaming Memory, you can have buffers in tile memory.
  Furthermore, you can choose whether to use a buffer in compute or IO tiles. Code
  can be copied in all directions between a compute tile buffer, IO tile buffer,
  and executable memory. This is done using :py:func:`ops.code_copy`.

  .. .. note:
  ..   :py:func:`ops.code_copy` is not yet implemented.

  Every graph has at most one compute buffer and one IO buffer created for it. As with
  the remote code buffers, these will be implicitly created if there is a single
  ``ops.code_copy`` that uses them, and all subsequent ``code_copy`` ops will reuse
  the same ones.

  You may wish to copy between locations on chip to temporarily free up memory
  in a location by exploiting the same kind of liveness analysis that was
  explained above. For example, you store the buffer in IO tiles, copy to a buffer
  in compute tiles, then copy back to the IO tiles buffer later. The final
  copy-back means the IO tile memory for the buffer can be reused in-between.

  When you copy into executable memory, there is no notion of a tileset that we are
  copying into. That is, a graph's code can have ops and tensors that span
  multiple tilesets, as it is the individual ops and tensors that are assigned
  tilesets, not the graph itself. The graph code will thus exist in the executable
  memory of whatever tiles Poplar deemed appropriate *across the chip*, not in one
  specific tileset, and so copying to executable memory can result in copies
  across the whole chip.

  You can also copy from Streaming Memory into these tile-memory buffers, using
  ``remote_code_load``. You may wish to do this for enabling overlapped IO. That
  is, you would use ``remote_code_load(g, destination='io_buffer')`` to copy from
  Streaming Memory into a buffer on IO tiles, whilst simultaneously performing
  computations on the compute tiles. Then, when required, you can copy the code
  back into executable memory using ``ops.code_copy``. Note there are further
  conditions for enabling IO overlap, which are explained in the next section.

  .. note
    :py:func:`ops.remote_code_load` to any destination other than ``executable``
    is not yet implemented.

  To summarise, you can copy from executable memory into a buffer on IO or compute
  tiles; copy between buffers in IO or compute tiles; and copy from IO or compute
  tiles into executable memory. All of these inter-IPU code copies are done
  with :py:func:`ops.code_copy`. You can also copy from Streaming Memory into a
  buffer on IO or compute tiles. All remote to IPU copies are done using
  :py:func:`ops.remote_code_load`.

Optimisation: merging the code load operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Poplar will attempt to merge exchanges of code data just like with other remote
buffer copies. That is, if you are loading code for multiple
graphs, and if there are no ops that cause a global exchange between the load ops in
the schedule (which you can ensure is the case using ``in_sequence``), then
Poplar will merge the exchanges for those loads, resulting in a speed-up. In
PopXL, it is up to you to decide if this is beneficial for your use-case
and impose such a schedule using :py:func:`in_sequence`.

Secondly, again as with regular tensors, careful scheduling of the ops can
ensure the IO of the code loading overlaps with computation. Though we cannot
give a full exposition of overlapped IO here, the basics are as follows: if
you want IO ``A`` to overlap with compute ``B``:

* ``A`` must come before ``B`` in the schedule.
* There can be no data-dependency between ``A`` and ``B``.
* ``A`` must be placed on IO tiles.
* ``B`` must be placed on compute tiles.
* If ``A`` consists of multiple stream copies, they must be adjacent in the Poplar sequence so that they are mergeable.

.. However, as stated in :ref:`Tile memory code buffers`, PopXL does not currently support code loading on
.. IO tiles, so overlapping it with compute is not currently possible.

Advanced example: nested code loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To help us further understand the semantics of code loading, let's examine
a nested example where we use ``remote_code_load`` to load a graph that
uses ``remote_code_load`` to load another graph:

.. literalinclude:: files/code_loading_nested.py
  :language: python
  :name: code_loading
  :caption: Example of nested code loading
  :start-after: BEGIN_BASIC
  :end-before: END_BASIC

.. only:: html

    :download:`Download code_loading.py <files/code_loading_nested.py>`

In this example, calling ``g1`` performs the load for ``g2``. After this, we can
now execute ``g2`` on device.

We could also change the ``load_g1`` function to instead take the ``Graph`` as a
parameter, then dynamically make many graphs for loading the code of other
graphs. Note however that the graph that performs the loading cannot
dynamically load any graph --- it is fixed to a certain graph on creation.
Only the function ``load_graph`` for creating such a ``Graph`` is dynamic and
can be reused for creating many graphs:

.. literalinclude:: files/code_loading_nested.py
  :language: python
  :name: code_loading
  :caption: More complex example of nested code loading
  :start-after: BEGIN_NOT_DYN
  :end-before: END_NOT_DYN

.. only:: html

    :download:`Download code_loading.py <files/code_loading_nested.py>`

Advanced concept: code loading in dynamic branches
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Graphs can have dynamic branching in them, for example through an ``if`` op.
Say there are ``ops.remote_code_load`` ops in these dynamic
branches, what effect will this have on the liveness of that code?

Liveness analysis is a static compile-time concept. We do not know which branch
will be taken at runtime. Say we perform the ``remote_code_load`` op in only one of
the branches, then call the graph after the branches merge again (so after
the ``if`` op). At the point of the call, the compiler does not know if the
``remote_code_load`` will have happened or not, as it does not know which branch
will be taken at runtime. The compiler has to produce a program that accounts
for all possible cases, so it must pessimistically plan as if the
``remote_code_load`` did not happen. Therefore, it will assume the code was
already live on the device before the branching.

Essentially, if there is branching before a ``call``, only if all possible
branches contain a ``remote_code_load`` can we assume that the code was dead and
in Streaming Memory until the ``remote_code_load`` op. If any possible branch does
not perform a ``remote_code_load``, we must assume that there was no
``remote_code_load`` and the code was already live before the branching.
