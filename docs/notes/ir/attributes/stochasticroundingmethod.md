# Developer Notes on Op Attribute 'stochasticRoundingMethod'

## High-level Description

Stochastic rounding is the process of rounding the result of an Op using
probabilistic behaviour. This is typically used when using lower-precision
floating point representations such as `fp16`. For more info on stochastic
rounding please read [this white
paper](https://docs.graphcore.ai/projects/ai-float-white-paper/en/latest/ai-float.html?highlight=stochastic%20rounding#deterministic-versus-stochastic-rounding).

PopART includes some logic to manage the case where stochastic rounding is used
in combination with replication. When replicating, you typically want:

* Stochastic rounding in the forward pass to be differing/distinct for each
  replica.
* Stochastic rounding in the weight update to be such that weights are keep in
  sync on each replica.

To achieve this, the random number state (RNG state, the value of
`poplar::getHwSeeds`) needs to be switched between these two parts of the
computational graph. This note explains the mechanisms involved in achieving
that.


## IR Representation

Every `popart::Op` has an optional 'stochastic rounding setting' attribute (e.g.
`Op::Settings::stochasticRoundingMethod`). The intent of this attribute is that
by the time the IR is lowered to Poplar this attribute it is set if and only if
`SessionOptions::enableStochasticRounding` is `true`.

In the case where stochastic rounding is enabled:

| Value                                             | Description                                                  |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `DifferingSeeds`        | Use this value when you want to apply stochastic rounding in a manner that is distinct on each replica. For example, gradient accumulation steps should use this setting. |
| `IdenticalSeeds`        | Use this setting when you want to apply stochastic rounding in a way that is guaranteed to result in identical rounding on all replicas. To achieve this, we reply on the invariant that the RNG state (i.e. the value of `poplar::getHwSeeds`) used for this setting must be the same value on all replicas prior to the Op executing. Use this option on, e.g., the optimizer's weight update step to ensure that the weight tensor on each replica applies the stochastic rounding in the same way and there is no weight drift. **NOTE: Ops that use this setting must adhere to [an invariant](#identicalseeds-invariant).** |

## IdenticalSeeds Invariant

To allow the outputs of `popart::Ops` that use `IdenticalSeeds` to stay 'in
sync' (i.e. be guaranteed to be identical) we rely on the following invariant:

* **Ops that use `StochasticRoundingMethod::IdenticalSeeds` *must* have the
  property that if the RNG state is identical on all replicas prior to executing
  the Op, then it must remain identical on all replicas after the execution of
  the Op has completed.**

  A typically sufficient (but not necessary) condition is that all input tensors
  of the Op have the same value across replicas.

## IR Assumptions

This section details the assumptions that the stochastic rounding logic places
on the PopART IR:

1. At time of lowering (when `IrLowering::prepareGraph` is called) the IR to
   poplar we assume that if `SessionOptions::enableStochasticRounding` is
   `false`, no Ops have `Op::Settings::stochasticRoundingMethod` set.
   Conversely, when `SessionOptions::enableStochasticRounding` is `true`, *all*
   Ops are expected to have their `stochasticRoundingMethod` set to *some*
   value.
2. At time of lowering the IR must be such that those Ops that use the setting
   `StochasticRoundingMethod::IdenticalSeeds` maintain [the
   invariant](#identicalseeds-invariant).

> **NOTE:** This assumption is verified by
> `StochasticRoundingAssumptionVerifier` at the end of `Ir::prepareIr` (but not
> currently checked for `popart.ir` IRs).

## Associated transformations

The `StochasticRounding` transform is responsible for assigning
`stochasticRoundingMethod` attributes in a way that is compatble with the
[assumptions](#ir-assumptions) outlined above. See
[stochasticrounding.md](../../transforms/stochasticrounding.md) for details.

## Lowering stochastic rounding logic

This section details how stochastic rounding logic is lowered to Poplar.

The `IrLowering` class has a member of type `RngStateLowering` that is
responsible for lowering code related to stochastic rounding. It has references
to two Poplar tensors which maintain the two RNG states:
```
  snap::Tensor differingSeedsRngStateTensor;
  snap::Tensor identicalSeedsRngStateTensor;
```

The `RngStateLowering` has functions to lower the following:

* `lowerInitRngStatesFromSeed` - Set both RNG states from a user's seed.
* `lowerSetRngState` - Set RNG state ready for an Op.
* `lowerGetRngState` - Get RNG state after the execution of an Op.

The latter two functions use the Op's `stochasticRoundingMethod` to determine
which RNG state to set/get.

The `IrLowering` class calls `lowerSetRngState` before growing each Op and
`lowerGetRngState` after calling each Op. In terms of what is lowered to Poplar
this may result in a sequence like this being lowered to Poplar (in
pseudo-code):
```
poplar::setHwSeeds(differingSeedsRngStateTensor)
// lowering of op0
differingSeedsRngStateTensor = poplar::getHwSeeds()
poplar::setHwSeeds(differingSeedsRngStateTensor)
// lowering of op1
differingSeedsRngStateTensor = poplar::getHwSeeds()
poplar::setHwSeeds(differingSeedsRngStateTensor)
// lowering of op2
differingSeedsRngStateTensor = poplar::getHwSeeds()
```
This is inefficient. We rely on a Poplar optimization to remove unnecessary
get/sets:
```
poplar::setHwSeeds(differingSeedsRngStateTensor)
// lowering of op0
// lowering of op1
// lowering of op2
differingSeedsRngStateTensor = poplar::getHwSeeds()
```
