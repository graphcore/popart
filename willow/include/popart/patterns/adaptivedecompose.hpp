// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_ADAPTIVEDECOMPOSE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_ADAPTIVEDECOMPOSE_HPP_

#include <vector>
#include <popart/patterns/optimizerdecompose.hpp>

namespace popart {
class Op;
class Tensor;

class AdaptiveDecompose : public OptimizerDecompose {
public:
  bool matches(Op *) const final;
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_ADAPTIVEDECOMPOSE_HPP_
