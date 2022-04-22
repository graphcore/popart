// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SGD0DECOMPOSE_PATTERN_HPP
#define GUARD_NEURALNET_SGD0DECOMPOSE_PATTERN_HPP

#include <vector>
#include <popart/patterns/optimizerdecompose.hpp>

#include "popart/names.hpp"

namespace popart {
class Graph;
class Op;
class SGD0ComboOp;
class Tensor;

/**
 * @brief Decomposes an SGD0ComboOp into the Ops and Tensors that implement the
 * SGD0 optimiser step it describes. \sa SGD0ComboOp \sa SGD.
 *
 * If gradient accumulation, will create the accum tensor (gradient
 * accumulator). This is not a persistent state tensor so will not be added to
 * the Ir's model proto. The tensor's id will have prefix
 * `reservedAccumPrefix()`. If the tensor has an initialiser in the model proto,
 * the tensor will be initialised with that value. Otherwise, it will be
 * initialised to 0. The DataType of the tensor is as specified in the
 * SGD0ComboOp.
 *
 * Recall the SGD0 optimiser step, possibly with gradient accumulation,
 * replication:
 *
 *   (_)    for each micro batch
 *   (1)      g = allReduce(g)       [if OptimizerReductionType=GradReduce]
 *   (2)      a += g                 [if grad acc]
 *   (_)    [let a := g if not grad acc]
 *   (3)    a = allReduce(a)         [if OptimizerReductionType=AccumReduce]
 *   (4)    w = (w * wdsf0) - (slr0 * a)
 *   (5)    a = 0                    [if grad acc]
 *
 * (1) is implemented by a ReplicatedAllReduceOp.
 *
 * (2) is implemented by an AccumulateOp.
 *
 * (3) is implemented by a ReplicatedAllReduceInplaceOp.
 *
 * (4) is implemented by an SGD0VarUpdateOp.
 *
 * (5) is implemented by an AccumulatorUpdateOp.
 *
 * For all the above ops, if they consume a non-const OptimizerValue, then the
 * SGD0ComboOp will have an additional input for that scalar, which will be
 * connected to the new Op.
 *
 * If gradient accumulation, (3-5) are put outside the microbatches loop
 * implicitly by setting
 *   op->settings.executionContext = ExecutionContext::AccumulateOuterFragment
 * Additionally, we will set
 *   op->settings.schedulePriority = 0.0f
 *   op->setExecutionPhases({})
 * because VarUpdateOps default to minimum possible schedule priority so they
 * are scheduled last, but this is not desirable for gradient accumulation, so
 * we reset to a neutral priority.
 *
 * The SGD0ComboOp will be disconnected and erased.
 */
class SGD0Decompose : public OptimizerDecompose {
public:
  bool matches(Op *) const final;
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;

  Op *varUpdateAndEraseCombo(Graph &graph,
                             SGD0ComboOp *combo,
                             const TensorId &weightId,
                             const TensorId &gradIntoUpdateId,
                             const TensorId &updatedWeightId) const;
};

} // namespace popart

#endif
