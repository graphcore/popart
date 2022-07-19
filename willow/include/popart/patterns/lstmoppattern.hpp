// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_LSTMOPPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_LSTMOPPATTERN_HPP_

#include <vector>
#include <popart/patterns/pattern.hpp>

namespace popart {
class Op;
class Tensor;

class LSTMPattern : public PreAliasPattern {
public:
  bool matches(Op *op) const override;

  std::vector<const Tensor *> touches(Op *) const override { return {}; }

  bool apply(Op *op) const override;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_LSTMOPPATTERN_HPP_
