.. _sec_transforms:


Transforms
==========

After an IR is built, you can use transforms or patterns to
manipulate its graphs in a non-trivial way.
Transforms are used to change a graph at the graph level, while
patterns are usually used to change a specific operation repeatedly
in a graph.

..  Applying transforms
  ^^^^^^^^^^^^^^^^^^^

  - Explain what transforms are available and how you use them.

Currently, we support the following transforms:

  -  Autodiff

..  -  Merge exchange

.. _autodiff:

Autodiff
-----------

In ``popart.ir`` you can use the ``autodiff`` function in the sub-package ``popart.ir.transforms``
to perform automatic differentiation on a per-graph basis. This transform creates a graph to compute
the gradients of a forward pass graph. It is declared as below:

.. code-block:: python

  autodiff(graph: Graph,
            grads_provided: Optional[Iterable[Tensor]] = None,
            grads_required: Optional[Iterable[Tensor]] = None,
            called_graphs_grad_info: Optional[Mapping[Graph, GradGraphInfo]] = None,
            return_all_grad_graphs: bool = False)

The first argument, ``graph``, is a forward pass graph. The ``grads_provided`` parameter indicates
for which outputs of ``graph`` we have gradients available for ``autodiff`` to use. For instance,
if ``graph`` outputs both loss and accuracy, you might not want to provide gradients for accuracy.
The ``grads_required`` indicates which inputs of the forward graph you want the ``autodiff`` function
to calculate gradients.  If not specified, the default ``grads_provided`` will be all of the
outputs of the forward graph and ``grads_required`` will be all of the inputs to the forward graph.
The arguments ``called_graphs_grad_info`` and ``return_all_grad_graphs`` are for nested graph.

.. TODO: Does this (below) mean *both* are needed if the graph contains calls to other subgraphs?
         But one or the other (or both?) might be needed in other cases? Or they are never needed in other cases?
         Or does it mean that one or the other is needed if the graph contains calls to other subgraphs?

They are both only needed if the graph you are applying autodiff to contains calls to other subgraphs.
The ``called_graphs_grad_info`` provides information required to apply ``autodiff`` to any calls from the
``graph`` to subgraphs. You can use it to customize the gradient graph when the autodiff graph does
not meet your needs. The ``return_all_grad_graphs`` indicates whether to return the gradient
graphs for all the graphs that recursively autodiff has been applied to or just for the given ``graph``.
The ``autodiff`` returns an ``GradGraphInfo`` object that includes the computational graph for
computing the gradients if the ``return_all_grad_graphs`` is set to ``False``. It will return all the
gradient graphs if the ``return_all_grad_graphs`` is set to ``True``.

The :py:class:`popart.ir.transforms.GradGraphInfo` object contains all the information and tools you need to call a gradient graph.

 -  ``graph``: the associated gradient graph as by `autodiff`
 -  ``forward_graph``: the forward graph that autodiff was applied to
 -  ``expected_inputs``: the tensors from the forward_graph that are required as inputs to the grad ``graph``
 -  ``expected_outputs``: the tensors from the forward_graph that have gradients as outputs of the grad ``graph``.
 -  ``inputs_dict(fwd_call_info)``: the inputs to call the gradient graph.
 -  ``fwd_graph_ins_to_grad_parent_outs(grad_call_info)``: the mapping between forward subgraph tensors and grad call site tensors. Note that the ``grad_call_info`` is the callsite info of the backward gradient graph.
 -  ``fwd_parent_ins_to_grad_parent_outs(fwd_call_info, grad_call_info)``: the mapping between forward call site inputs and grad call site outputs. It can be used to get the gradient with respect to a specific input.

You can then call the gradient graph returned by autodiff to calculate the required gradients.
The partial derivatives of the loss with respect to the graph outputs of the forward graph are
the first inputs of the gradient graph. The following example shows how to calculate the gradients
with ``autodiff`` for a ``linear_graph``.

#. First operation ``call_with_info`` returns ``fwd_call_info`` that contains the callsite info.
#. Then get the gradient graph ``bwd_graph_info`` by using ``autodiff`` on the ``linear_graph``.
#. Then get all the activations calculated in the forward pass with the backward graph using ``bwd_graph_info.inputs_dict`` with the ``call_info`` of the forward call as an argument.
#. Last, call the gradient graph using ``ops.call``. The argument ``grad_seed`` is the initial value of the partial gradient. Increasing this ``grad_seed`` can serve as loss scaling. The ``activation`` is used to connect the input of the gradient graph with the caller graph.


.. literalinclude:: ../user_guide/files/autodiff_popart_ir.py
  :language: python
  :start-after: Op begin
  :end-before: Op end

.. only:: html

    :download:`Download autodiff_popart_ir.py <../user_guide/files/autodiff_popart_ir.py>`
