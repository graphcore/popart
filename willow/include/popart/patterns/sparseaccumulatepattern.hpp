// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSPOSE_TO_IDENTITY_OR_RESHAPE_PATTERN_HPP
#define GUARD_NEURALNET_TRANSPOSE_TO_IDENTITY_OR_RESHAPE_PATTERN_HPP

#include <popart/patterns/pattern.hpp>

namespace popart {

/**
 * @brief Replaces GatherGrad -> Accumulate with a single SparseAccumulate.
 *
 * Replaces
 *   dX -> GatherGrad -> dW -> Accumulate(w) -> updatedW
 * With
 *   dX -> SparseAccumulate(w) -> updatedW
 *
 * Removing the intermediate dense tensor dW.
 */
class SparseAccumulatePattern : public PreAliasPattern {
public:
  bool matches(Op *op) const override;

  std::vector<const Tensor *> touches(Op *) const override;

  bool apply(Op *op) const override;
};

} // namespace popart

#endif
