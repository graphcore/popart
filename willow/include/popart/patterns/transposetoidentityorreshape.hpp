// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSPOSE_TO_IDENTITY_OR_RESHAPE_PATTERN_HPP
#define GUARD_NEURALNET_TRANSPOSE_TO_IDENTITY_OR_RESHAPE_PATTERN_HPP

#include <vector>

#include "popart/patterns/pattern.hpp"

namespace popart {
class Op;
class Tensor;

// Replaces:
//  Transpose -> {Identity/Reshape} -> Transpose
//          with
//  Identity/Reshape

class TransposeToIdentityOrReshapePattern : public PreAliasPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
