// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ADAPTIVEDECOMPOSE_PATTERN_HPP
#define GUARD_NEURALNET_ADAPTIVEDECOMPOSE_PATTERN_HPP

#include <popart/patterns/optimizerdecompose.hpp>
#include <popart/patterns/patterns.hpp>

namespace popart {

class AdaptiveDecompose : public OptimizerDecompose {
public:
  bool matches(Op *) const final;
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;
};

} // namespace popart

#endif
