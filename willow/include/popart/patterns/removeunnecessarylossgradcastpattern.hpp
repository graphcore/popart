// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ALS_REMOVE_CLIPPING_PATTERN_HPP
#define GUARD_NEURALNET_ALS_REMOVE_CLIPPING_PATTERN_HPP

#include <vector>

#include "popart/patterns/pattern.hpp"

namespace popart {
class Op;
class Tensor;

/*!
 * The RemoveUnnecessaryLossGradCast changes
 *
 * \code
 * fp32_lossScale -- Cast -- fp16_lossScale -- NllLossGradOp -- fp16_grad
 *                           fp16_probs -------'
 * \endcode
 *
 * to
 *
 * \code
 * fp32_lossScale -- NllLossGradOp -- fp16_grad
 * fp16_probs -------'
 * \endcode
 *
 * This corner case can occur in a model with fp16 activations when its fp16
 * loss scale is anchored for summation and upcast to fp32 in order to prevent
 * overflow. In this case if we have a loss scale >max(fp16) the downcasting
 * will result in a clipping of the loss scale.
 *
 * Notice that even if the loss scale is >max(fp16) the resulting gradients can
 * be within fp16 range. If the resulting gradients are >max(fp16), they will be
 * clipped (unless the user has enabled \c NaN on overflow).
 */
class RemoveUnnecessaryLossGradCast : public PreAliasPattern {
public:
  bool matches(Op *lossOp) const final;
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *lossOp) const final;
};

} // namespace popart

#endif
