# Transform Developer Notes

| ****      |                                                                  |
| --------- | ---------------------------------------------------------------- |
| **NOTE**: | These notes are a work in progress, please amend as you see fit. |
|           |                                                                  |

A transform is an object that can change the [PopART IR](../ir.md) in some
non-trivial way. Transforms are distinct from patterns in that transforms are
typically applied only once (or a few times) as opposed to repeatedly in a
fix-point. Also, transforms typically transform the [PopART IR](../ir.md) in a
way that is more involved than replacing one `Op` with some other ops.

## List of existing transforms:

<mark>NOTE: This section is incomplete.</mark>

Please find below a brief description of existing transforms:

| Transform                                                         | Description                                                                                                                                          |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| AccumulateOuterFragment-<br/>Parallizer                           | Add topological constraints to accumulate outer fragment to parallelise weight updates.                                                              |
| AutoVirtualGraph                                                  | Associate each `Op` in the main graph with a virtual graph.                                                                                          |
| Autodiff                                                          | Construct the backwards pass, adding gradient tensors for model parameters where possible.                                                           |
| AutomaticLossScale                                                | Adds operations that automatically manage a loss scaling factor.                                                                                     |
| BatchSerialize(1)                                                 | Split and serialise chains of `Ops` in the batch dimension.                                                                                          |
| BatchSerialize(2)                                                 | Adjust the batch serialized `Ops` schedule.                                                                                                          |
| ClipWeightGradientsByNorm                                         | <mark>TBD</mark>                                                                                                                                     |
| DecomposeGradSum                                                  | Replacing gradient `SumOps` with series of `AddOps` to reduce tensor liveness.                                                                       |
| DecomposeLoops                                                    | Unroll `LoopOp` iterations to enable overlap between IO and compute tiles.                                                                           |
| DynamicOp                                                         | Replace dynamic grad operations with canonical dynamic update/add/slice operations.                                                                  |
| ExplicitRecompute                                                 | Clone main graph forward pass operations into the backward pass to reduce tensor liveness.                                                           |
| GroupMatMuls                                                      | Combine multiple `MatMulOps` into a single `Op`.<br/>(**NOTE**: Deprecated, to be removed after release).                                            |
| HostIoSetup(1)                                                    | Create a `InitOp`->`HostLostOp`->input combo for each input tensor (stream from host) tensor.                                                        |
| HostIoSetup(2)                                                    | Add a `HostStoreOp` (no init) for each output tensor (stream to host).                                                                               |
| HostReduce                                                        | <mark>TBD</mark>                                                                                                                                     |
| InferPipelineStages                                               | If no nodes in the Onnx model have the 'PipelineStage' attribute set, assign pipeline stages to `Ops` based on their associated virtual graphs.      |
| InplaceAccumulateGrad-<br/>PartialsIntoOptimizer-<br/>AccumTensor | Reduce tensor liveness by directly accumulating grad partials into the optimizer accumulator instead of a separate gradsum accumulator first.        |
| InterIpuCopy                                                      | Make explicit any copying of tensor data between IPUs (by adding `IpuCopyOps`).                                                                      |
| IoComputeTileCopy                                                 | Make explicit any copying of tensor data between IO and compute tiles on one IPU (by adding `IoTileCopyOp`).                                         |
| MainLoops                                                         | Make the main graph training loop an explicit `LoopOp`.                                                                                              |
| MergeCopies                                                       | Combine groups of `IpuCopyOps` that are consumed by the same `Op`.                                                                                   |
| MergeDuplicateOps                                                 | Combine `Ops` that perform the exact same computation.                                                                                               |
| MergeLoops                                                        | Combine compatible `LoopOps` into one.                                                                                                               |
| MergeRemote                                                       | Combine adjacent `RemoteLoadOps`/`RemoteStoreOps`/`RemoteExchangeOps` into RemoteExchange operations for efficiency.                                 |
| MergeVarUpdates                                                   | Combine compatible `VarUpdateOp` ops.                                                                                                                |
| Pipeline                                                          | Remove dependencies between `Ops` assigned to different pipeline stages on the same IPU by adding `StashOps` and `RestoreOps` to the graph.          |
| PreAutomaticLossScale                                             | Annotate a user-specified list of tensors by passing them through an `AutomaticLossScalingProxyOp`, to find their gradients in `AutomaticLossScale`. |
| Prune                                                             | Remove unused tensors and `Ops` from a specific graph.                                                                                               |
| RandomSetup                                                       | Add random seed inputs to any `Ops` that require them.                                                                                               |
| RemoteSetup                                                       | Assign remote buffer identities and offsets to RemoteArg tensors.                                                                                    |
| SerializeMatMuls                                                  | Split `MatMulOps` over a given dimension in order to serialise and make memory usage more granular.                                                  |
| StreamingMemory(1)                                                | Map ops to phases, enable caching on variables.                                                                                                      |
| StreamingMemory(2)                                                | Enable caching on variables, map remaining ops to phases, cut graph and insert replicated tensor sharding, remote load and remote store ops.         |
| SubgraphOutline                                                   | Extract repeated `Ops` structures into new `Graphs` and call them with `CallOps` to safe memory.                                                     |
|                                                                   |

## Assumptions & guarantees for existing transforms:

For each of the transforms above, we document assumptions and guarantees related
to the state of the PopART IR for each transform. In this context, with
**assumption** we mean a pre-condition. That is, something that *must* hold
before the transform is applied. Analogously, a **guarantee** is a
post-condition and is something that *will* hold after a transform is applied,
provided all assumptions are met beforehand. Finally, a transform **preserves**
a condition if applying the transform does not make the condition become untrue.

Please find below a list of conditions that we use to describe assumptions and
guarantees of transforms:

| State              | Description                                                                                             |
| ------------------ | ------------------------------------------------------------------------------------------------------- |
| `NO_INPLACE`       | There are no inplace `Ops` in any `Graph`.                                                              |
| `NO_IPU_COPY`      | There are no `IpuCopyOps` in any `Graph`.                                                               |
| `NO_IMPLICIT_COPY` | No `Op` in any `Graph` is on a different IPUs than the tensors that they consume (except `IpuCopyOps`). |
| `VGRAPH`           | Each `Op` in the main graph has a `VGraphId` assigned to it.                                            |
| `PSTAGE`           | Each `Op` in the main graph has a `PipelineStage` assigned to it.                                       |
| `EPHASE`           | Each `Op` in the main graph has an `ExecutionPhase` assigned to it.                                     |
| `BPHASE`           | Each `Op` in the main graph has a `BatchSerializedPhase` assigned to it.                                |
|                    |                                                                                                         |

We refer to these conditions in our list of assumptions, guarantees and preserved conditions below:

| Transform                                                         | Assumptions                     | Guarantees                      | Preserved conditions |
| ----------------------------------------------------------------- | ------------------------------- | ------------------------------- | -------------------- |
| AccumulateOuter-<br/>FragmentParallizer                           |                                 |                                 |                      |
| AutoVirtualGraph                                                  |                                 | `VGRAPH`                        |                      |
| Autodiff                                                          | `NO_INPLACE`,<br/>`NO_IPU_COPY` | `NO_INPLACE`,<br/>`NO_IPU_COPY` |                      |
| AutomaticLossScale                                                |                                 |                                 |                      |
| BatchSerialize                                                    |                                 |                                 |                      |
| ClipWeightGradientsByNorm                                         |                                 |                                 |                      |
| DecomposeGradSum                                                  |                                 |                                 |                      |
| DecomposeLoops                                                    |                                 |                                 |                      |
| DynamicOp                                                         |                                 |                                 |                      |
| ExplicitRecompute                                                 |                                 |                                 |                      |
| GroupMatMuls                                                      |                                 |                                 |                      |
| HostIoSetup                                                       |                                 |                                 |                      |
| HostReduce                                                        |                                 |                                 |                      |
| InferPipelineStages                                               | `VGRAPH`                        | `PSTAGE`                        |                      |
| InplaceAccumulateGrad-<br/>PartialsIntoOptimizer-<br/>AccumTensor |                                 |                                 |                      |
| InterIpuCopy                                                      |                                 | `NO_IMPLICIT_COPY`              |                      |
| IoComputeTileCopy                                                 |                                 |                                 |                      |
| MainLoops                                                         |                                 |                                 |                      |
| MergeCopies                                                       |                                 |                                 |                      |
| MergeDuplicateOps                                                 |                                 |                                 |                      |
| MergeLoops                                                        |                                 |                                 |                      |
| MergeRemote                                                       |                                 |                                 |                      |
| MergeVarUpdates                                                   |                                 |                                 |                      |
| Pipeline                                                          | `PSTAGE`                        |                                 |                      |
| PreAutomaticLossScale                                             |                                 |                                 |                      |
| Prune                                                             |                                 |                                 |                      |
| RandomSetup                                                       |                                 |                                 |                      |
| RemoteSetup                                                       |                                 |                                 |                      |
| SerializeMatMuls                                                  |                                 |                                 |                      |
| StreamingMemory                                                   |                                 |                                 |                      |
| SubgraphOutline                                                   |                                 |                                 |                      |
|                                                                   |                                 |                                 |                      |


## Things to consider (when implementing a transform):

<mark>NOTE: This section is incomplete.</mark>

* **Tensor naming** It is important that any tensors a transform introduces are
  named appropriately, as PopART using names to work out some properties of
  tensors. For example, if you add an accumulator tensor, make sure you use the
  appropriate prefix (see `tensornames.hpp`):

  ```c++
  TensorId myTensor = reservedAccumPrefix() + "foo";
  graph.getTensors().addVarInit(myTensor, ...);
  ```

* **Determinism** The transformation's changes should be deterministic. That is,
  for a two Graph objects that are equivalent in terms of values, the transform
  should make equivalent changes to both graphs. Common ways in which
  non-determinism is introduced is by accidentally using containers ordered by
  pointer value. For example, using a `std::map` with a `key_type` of `Op*` and
  iterating over entries of this map type results in a non-determinism iteration
  order:

  ```c++
  std::map<Op*, Foo> fooMap;         // bad!
  std::map<Op*, Foo, POpCmp> fooMap; // okay!
  ```

* **Graph reachability** You should ensure your transform doesn't make `Graphs`
  in the IR unreachable from the main graph (this would break, e.g., the
  `LivenessAnalyzer`). We loosely borrow the definition of reachability from
  imperative programming:

  * An `Op` reaches the graphs returned by the
    `Op::getCalledGraphs()` function.
  * A `Graph` reaches the union of graphs reached by the `Ops` it contains.

  A `Graph` that is unreachable from the main graph is akin to a function that
  is never called by an application. Hence, another way of phrasing this
  requirement is: *don't introduce dead code*.

* **Graph inputs/outputs** Ops with called graphs rely on an
  `Op`-specific relation between the `Op`'s inputs/outputs and the called
  `Graph`'s inputs/outputs. Hence, transforms need to be careful when modifying
  the inputs/outputs of a `Graph` and will have to adapt any 'call site' `Op`
  accordingly when they do modify them.

* **Setting attributes** To make a transform as widely applicable as possible
  (see later sections) ensure that if all `Ops` in the main graph have a
  `VGraphId` (resp. `PipelineStage`, `ExecutionPhase`, `BatchSerializedPhase`)
  associated with it *before* the tranform is applied then make sure this is
  also the case *afterwards*. You can do this by never introducing an `Op` in
  your transform without also inheriting the `VGraphId` (resp. `PipelineStage`,
  `ExecutionPhase`, `BatchSerializedPhase`) from a nearby `Op`:

  ```c++
  auto op = graph.createConnectedOp<FooOp>(...);
  op->inheritPlacementAttributes(true);
  op->setup();
  ```
  
  **NOTE**: In other words, make sure the transform preserves the `VGRAPH`,
  `PSTAGE`, `EPHASE` and `BPHASE` as defined above. If these conditions hold
  before the transform is applied, then they should hold afterwards, also.
  
* **Aliases** If a transform modifies, adds or removes any tensor or `Op` that
  is aliasing, the aliases must be updated after the transform has finished. If the
  transform itself relies on aliasing information, it can also update the aliases
  during the transform:
  
  ```c++
  if (getSessionOptions().enableSerializedMatmuls) {
    applyTransform(SerializeMatMuls::id(), getMainGraph());
    // SerializeMatMuls could have changed aspects of aliasing
    updateAliases();
    updateVertices();
  }
  ```
