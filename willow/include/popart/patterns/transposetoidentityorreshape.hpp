// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TRANSPOSE_TO_IDENTITY_OR_RESHAPE_PATTERN_HPP
#define GUARD_NEURALNET_TRANSPOSE_TO_IDENTITY_OR_RESHAPE_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

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