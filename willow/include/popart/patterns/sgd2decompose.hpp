// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD2DECOMPOSE_PATTERN_HPP
#define GUARD_NEURALNET_SGD2DECOMPOSE_PATTERN_HPP

#include <vector>
#include <popart/patterns/optimizerdecompose.hpp>

#include "popart/names.hpp"

namespace popart {

class Graph;
class Op;
class SGD2ComboOp;
class Tensor;

/**
 * @brief Decomposes an SGD2ComboOp into the Ops and Tensors that implement the
 * SGD2 optimiser step it describes. \sa SGD2ComboOp \sa SGD.
 *
 * Will create the accl1 tensor (first-order momentum). The tensor will be added
 * to the Ir's model proto, so will be part of the serialised Ir. The tensor's
 * id will have prefix `reservedAccl1Prefix()`. The tensor will be initialised
 * to 0.The DataType of the tensor is as specified in the SGD2ComboOp.
 *
 * If gradient accumulation, will create the accum tensor (gradient
 * accumulator). This is not a persistent state tensor so will not be added to
 * the Ir's model proto. The tensor's id will have prefix
 * `reservedAccumPrefix()`. If the tensor has an initialiser in the model proto,
 * the tensor will be initialised with that value. Otherwise, it will be
 * initialised to 0. The DataType of the tensor is as specified in the
 * SGD2ComboOp.
 *
 * Recall the SGD2 optimiser step, possibly with gradient accumulation,
 * replication:
 *
 *   (_)    for each micro batch
 *   (1)      g = allReduce(g)       [if OptimizerReductionType=GradReduce]
 *   (2)      a += g                 [if grad acc]
 *   (_)    [let a := g if not grad acc]
 *   (3)    a = allReduce(a)         [if OptimizerReductionType=AccumReduce]
 *   (_)    // Note we break the single v
 *             update equation into two steps:
 *   (4)    v += dpsf1 * a
 *   (5)    v = v * smm1 + swd1 * w
 *   (6)    w = w - slr1 * v
 *   (7)    a = 0                    [if grad acc]
 *
 * See the SGD docs in optimizer.hpp for derivation of the above.
 *
 * (1) is implemented by a ReplicatedAllReduceOp.
 *
 * (2) is implemented by an AccumulateOp.
 *
 * (3) is implemented by a ReplicatedAllReduceInplaceOp.
 *
 * (4) is implemented by an AccumulateOp.
 *
 * (5) is implemented by an SGD2AcclUpdateOp. Note this is equivalent to an
 * SGD1AcclUpdateOp.
 *
 * (6) is implemented by an SGD2VarUpdateOp. Note this is equivalent to an
 * SGD1VarUpdateOp.
 *
 * (7) is implemented by an AccumulatorUpdateOp.
 *
 * For all the above ops, if they consume a non-const OptimizerValue, then the
 * SGD2ComboOp will have an additional input for that scalar, which will be
 * connected to the new Op.
 *
 * If gradient accumulation, (3-8) are put outside the microbatches loop
 * implicitly by setting
 *   op->settings.executionContext = ExecutionContext::AccumulateOuterFragment
 * Additionally, we will set
 *   op->settings.schedulePriority = 0.0f
 *   op->setExecutionPhases({})
 * because VarUpdateOps default to minimum possible schedule priority so they
 * are scheduled last, but this is not desirable for gradient accumulation, so
 * we reset to a neutral priority.
 *
 * The SGD2ComboOp will be disconnected and erased.
 *
 * Additional topo cons are required to ensure the VarUpdateOps run in the
 * manner described above. We also must transfer the topo cons from the
 * SGD2ComboOp to the new ops without breaking this. To do this:
 *
 *  1. Transfer topo cons from combo to (1).
 *  2. Transfer topo cons from combo to (2).
 *  3. Insert topo con from (6) to (8) to ensure accum not zeroed until after
 *     v update (which consumes it).
 *  4. Transfer topo cons from combo to (7). Only required if not grad acc.
 */
class SGD2Decompose : public OptimizerDecompose {
public:
  bool matches(Op *) const final;
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;

private:
  TensorId acclUpdate(Graph &graph,
                      const SGD2ComboOp *combo,
                      const TensorId &gradIntoAcclId,
                      const TensorId &accl1Id,
                      const TensorId &weightId) const;

  void varUpdateAndEraseCombo(Graph &graph,
                              SGD2ComboOp *combo,
                              const TensorId &weightId,
                              const TensorId &updatedAcc1lId,
                              const TensorId &updatedWeightId) const;
};

} // namespace popart

#endif
