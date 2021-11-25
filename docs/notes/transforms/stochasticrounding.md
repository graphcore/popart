# Developer Notes on the Stochastic Rounding Transform

## Context

The stochastic rounding transform is responsible for assigning a
`stochasticRoundingMethod` attribute value to every Op in the IR.

> **NOTE:** You should *only* run this transform when
> `SessionOption::enableStochasticRounding` is set so as to not violate the
> assumption that holds on the `stochasticRoundingMethod` attribute at lowering
> time (see
> [stochasticroundingmethod.md](../ir/attributes/stochasticroundingmethod.md)
> for details on this invariant).

> **NOTE:** To ensure that *every* Op is assigned a `stochasticRoundingMethod`
> value at the end of `Ir::prepareImpl` the stochastic rounding transform is
> currently the last transform that is ran. This is because, at time of writing,
> other transforms do not set the `stochasticRoundingMethod` when they create
> new Ops, so running those transforms *after* the stochastic rounding transform
> would lead to having Ops that do not have the attribute set at time of
> lowering (and this violates the assumption set out in
> [stochasticroundingmethod.md](../ir/attributes/stochasticroundingmethod.md)).

> **NOTE:** While still under test, this transform must be explicitly enabled by
> setting the temporary session option
> `SessionOptions::_enableRngStateManagement` to `true`. By default, this option
> is `false`. When `false`, the transform will simply assign all Ops the
> `stochasticRoundingMethod` value of `IdenticalSeeds`. This is to preserve
> existing behaviour while the new functionality is being tested. As detailed in
> T48752, the plan is to remove this session option and unconditionally enable
> the proper behaviour of the stochastic rounding transform once it's been
> tested.
>
> This note should be removed as part of completing T48752.

The stochastic rounding transform does the following:

1. It performs a `replica-equal` analysis to establish for Op input (and output)
   whether the respective tensor is guaranteed to be identical in value on all
   replicas at the time the input is consumed (or output produced,
   respectively).
2. Iterate over all Ops and set the `stochasticRoundingMethod` attribute to
   `IdenticalSeeds` if all outputs of said op are determined to be
   replica-equal, and setting it to `DifferingSeeds` otherwise.

The remainder of these notes describes the replica-equal analysis described
above.

## Replica-Equal Analysis

### Modelling PopART Semantics

In the PopART IR, a single tensor attains one or more values over time. There
are things to note here:

* `TensorType::Const` tensors attain a value once, at initialisation time, and
  never change.
* `TensorType::Stream` tensors are used for streaming various things (input
  data, random seeds, etc.) to the device. These tensors can attain values many
  times, but need not be produced or modified by an Op (see
  `SessionOptions::useHostCopyOps`).
* `TensorType::Variable` tensors are given a value at initialisation time but
  their value typically is modified over time by Ops that modify their inputs
  where this tensor is said input (or an alias of said input).
* `TensorType::ActGrad` tensors attain a value when they are produced, but also
  may have their value modified over time. This happens when an Op modifies an
  input and the `ActGrad` is connected to that input. It can also be that a
  different tensor is connected to the input but that tensor aliases the
  `ActGrad`.

> **NOTE:** Both inplaced Ops (e.g. `SGD0VarUpdateOp`, `ScaleInplaceOp`) and Ops
> like `CallOp` can modify their input tensors. As an aside, some of these Ops
> may output an `ActGrad` tensor that aliases some or all of their modified
> inputs, but that isn't universally the case.

Because tensor values (and hence the replica-equal property of said tensor) may
change over time (due to ops modifying their inputs), the replica-equal analysis
does not assign a replica-equal value to tensors, but to *tensor values*.

> **NOTE:** We identify *tensor values* by a tuple (`Tensor`, `int`) where
> `Tensor` is a `const popart::Tensor*` pointer and the `int` is a *time* value
> that relates to the index of the Op that either *produces* or *modifies* in a
> graph schedule for the graph in which the tensor resides. Tensor values that
> are assigned at initialisation time (and hence not associated with an Op) are
> given a time of `-1`.

The replica-equal analysis' job is to calculate a mapping from tensor values to
a `bool` values. The aim is for this mapping to accurately contain a value for
all tensors in the IR such that for every tensor, we have that it's
replica-equal value is `true` when the tensor is replica-equal and `false`
otherwise. However, where inaccuracy is unavoidable, the analysis errs on the
side of not wrongly identifying tensors as replica-equal where they may not be.

### Forward Propagation

The replica-equal analysis first deals with tensor values that are given at
initialisation time:

1. For tensors with initial values we do the following:

   * The initial (and only) value of `TensorType::Const` tensors is assumed
     replica-equal (except for random seed used to explicitly seed random ops).
   * The initial value (and all subsequent values) of `TensorType::Stream`
     tensors that are not produced by `HostLoadOps` are assumed replica-equal if
     their stream mode is `ReplicatedStreamMode::Broadcast` and not
     replica-equal otherwise.
   * The initial value of `TensorType::Variable` tensors are assumed
     replica-equal (more on this later).

   > **NOTE:** Graph inputs also have initial values, but these are expanded
   > by the analysis when it expands a subgraph Op (e.g. `CallOp`).

2. Then, starting with the main graph, we propagate replica-equal values through
   ops, following a valid schedule order. The Op order is relevant because we
   need an up-to-date replica-equal value for each Op input before replica-equal
   values are propagated through an Op. The forward propagation is implemented
   per-Op in the virtual function `Op::fwdPropagateIsReplicaEqual` which is a
   function with the following prototype:

   ```c++
   virtual std::tuple<ReplEqOutputMap, ReplEqModifiedInputMap>
   fwdPropagateIsReplicaEqual(const AliasModel &aliasModel,
                              const ReplEqInputMap &inputMap,
                              ReplicaEqualAnalysisProxy &proxy) const;
   ```
   where `aliasModel` is an up-to-date alias model that contains alias
   information pertaining to the graph the Op is in, `inputMap` is a mapping
   from input indices to replica-equal values and `proxy` is an object that
   contains contains methods to, e.g., propagate replica-equal values through
   subgraph (used by, e.g., `CallOp`).

   An Op returns both a mapping from output indices to replica-equal values for
   outputs and a mapping from input indices to replica-equal values for inputs
   that are modified.

   The `Op::fwdPropagateIsReplicaEqual` function is implemented by Ops as
   followed:

   * By default an Op's output is replica-equal if and only if all inputs are
     replica-equal.
   * `ReplicatedAllGatherOp` output is always replica-equal.
   * `ReplicatedAllReduceOp` output is always replica-equal.
   * `ReplicatedReduceScatterOp` output is never replica-equal.
   * `HostLoadOp` output is replica-equal if and only if the stream mode is
     `ReplicatedStreamMode::Broadcast`.
   * `MultiExchangeOp` outputs associated with a host exchange in the load
     direction are replica-equal if and only if the associated stream mode is
     `ReplicatedStreamMode::Broadcast`. Outputs associated with a remote
     exchange in the load direction are replica-equal if and only if all inputs
     associated with the same remote exchange are replica-equal.
   * `CallOp`, `LoopOp` and `IfOp` map the replica-equal value of their Op
     inputs to replica-equal values for the graph inputs of the subgraphs that
     they call. Then, they propagate the replica-equal values through their
     respective subgraphs. Finally they map the replica-equal values for the
     graph outputs to back to the Op's outputs. For `IfOp`, which has two
     subgraphs, we say the Op's output is replica-equal if and only if both
     subgraph outputs agree that this is the case.

     > **NOTE:** `LoopOp` support is currently experimental.

### Repeat Until Fixpoint

There is a slight complication with our basic algorithm which is that it is not
possible to reliably determine from the IR whether variable tensors are
replica-equal. What we do for those tensors is to initially assume that they are
replica-equal and, if at some point they are updated with a not replica-equal
value, then we consider them not replica-equal from that point onwards.

More concretely, we initially assign a replica-equal value of `true` to variable
tensors. Then, we do a full forward propagation of replica-equal values through
the main graph. As part of this propagation we check if any variable tensors got
assigned a value that is not replica-equal. If so, we set that variable to
`false`.

> **NOTE:**: We do this check in
> `ReplicaEqualAnalysisImpl::processMainGraphAliases()`.

> **NOTE:** For a variable to be assigned a value that is not replica-equal an
> Op in the main graph must *modify* an input tensor that aliases the variable
> tensor in such a way that the modified value is not replica-equal.

If a variable changes values then we repeat the forward propagation, because
this change might affect the results of the analysis. We repeat this process
until there are no further changes.

Formally speaking this makes the analysis a fixpoint computation and we are
computing a greatest fixpoint (assuming an order where replica equal is less
than replica equal).

> **NOTE:**: This fixpoint computation is done in
> `ReplicaEqualAnalysisImpl::apply()`.

## Check for disagreements between subgraph callsites
The description above glosses over one detail: there is no guarantee that, when
subgraphs are called from multiple call sites, that those call sites will agree
on the replica-equal values for said subgraph's inputs. Recall that for graph
inputs, the replica-equal value of the graph input is the logical AND over all
call sites, so if there is disagreement, the value of the input is `false`.

To warn the user about this, once we reach a fixed point, we do one more
iteration and look for any tensors that propagated a `true` value whilst already
having been marked as being `false`. They will not change value, because of the
logical AND, but we do warn the users about these tensors. Generally, this
kind of disagreement on value should only happen for graph inputs.

> **TODO(T48510):** The outlining transform may introduce disagreements
> between call sites and must be modified to explicitly avoid this by running
> replica-equal analysis and not outlining Ops together that disagree on
> replica-equalness.
>
> This note should be removed as part of completing T48510.

> **TODO(T51589):** The currently replica-equal analysis is quite coarse-grained
> in that we mark tensor values as either being replica-equal (`true`) or not
> (`false`). When using collectives with comms groups, it is possible for these
> values to be replica-equal for some but not all replicas. A more fine-grained
> analysis could account for this and would allow management of RNG states
> accordingly.
>
> This note should be removed as part of completing T48510(T51589).
