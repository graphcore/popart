// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_SGD1DECOMPOSE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_SGD1DECOMPOSE_HPP_

#include <vector>
#include <popart/patterns/optimizerdecompose.hpp>

namespace popart {
class Op;
class Tensor;

/**
 * @brief Decomposes an SGD1ComboOp into the Ops and Tensors that implement the
 * SGD1 optimiser step it describes. \sa SGD1ComboOp \sa SGD.
 *
 * Will create the accl tensor (combined accumulator and first-order momentum).
 * The tensor will be added to the Ir's model proto, so will be part of the
 * serialised Ir. The tensor's id will have prefix `reservedAcclPrefix()`. If
 * the tensor has an initialiser in the model proto, the tensor will be
 * initialised with that value. Otherwise, it will be initialised to `slr1 *
 * w_0`, where w_0 is the initial value of w.
 *
 * Recall the SGD1 optimiser step, possibly with gradient accumulation and
 * replication:
 *
 *   (_)   for each micro batch
 *   (1)     allReduce(g)          [if OptimizerReductionType=GradReduce]
 *   (2)     v += dpsf1 * g
 *   (_)     if enable nesterov momentum:
 *   (_)       a += g
 *   (3)   v = allReduce(v)        [if OptimizerReductionType=AcclReduce]
 *   (_)   if enable nesterov momentum:
 *   (_)     a = allReduce(a)      [if OptimizerReductionType=AcclReduce]
 *   (4)   if enable nesterov momentum:
 *           ils = ndsf * dpsf1
 *           a = ngsf * (ils * a + wd * w) + mm * v
 *   (_)   [let x := g if enable nesterov momentum else v]
 *   (5)   w = w - slr1 * x
 *   (6)   v = v * smm1 + swd1 * w
 *
 * See the SGD docs in optimizer.hpp for derivation of the above.
 *
 * (1) is implemented by a ReplicatedAllReduceOp.
 *
 * (2) is implemented by an AccumulateOp.
 *
 * (3) is implemented by a ReplicatedAllReduceInplaceOp.
 *
 * (4) is implemented by a MulOp and a SGD1NesterovOp.
 *
 * (5) is implemented by an SGD1VarUpdateOp.
 *
 * (6) is implemented by an SGD1AcclUpdateOp.
 *
 * For all the above ops, if they consume a non-const OptimizerValue, then the
 * SGD1ComboOp will have an additional input for that scalar, which will be
 * connected to the new Op.
 *
 * If gradient accumulation, (3), (4), (5), (6) are put outside the microbatches
 * loop implicitly by setting
 *   op->settings.executionContext = ExecutionContext::AccumulateOuterFragment
 * Additionally, we will set
 *   op->settings.schedulePriority = 0.0f
 * because VarUpdateOps default to minimum possible schedule priority so they
 * are scheduled last, but this is not desirable for gradient accumulation, so
 * we reset to a neutral priority.
 *
 * The SGD1ComboOp will be disconnected and erased.
 *
 * Additional topo cons are required to ensure the VarUpdateOps run in the
 * manner described above. We also must transfer the topo cons from the
 * SGD1ComboOp to the new ops without breaking this. To do this:
 *
 *  1. At the start of apply, add a topo con from (1) to the combo op.
 *  2. Transfer topo cons from combo to (2). Since (1)/(2) are the first op to
 *     run in the optimiser step (the other ops consume (2)'s output so will
 *     always run after), this ensures the pre-existing topo cons on combo are
 *     respected.
 *  3. Insert topo con from (5) to (6), to ensure w update happens before the
 *     next step's v update.
 */
class SGD1Decompose : public OptimizerDecompose {
public:
  bool matches(Op *) const final;
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_SGD1DECOMPOSE_HPP_
