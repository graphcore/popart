// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_NEGATIVE_ONE_SCALE_PATTERN_HPP
#define GUARD_NEURALNET_NEGATIVE_ONE_SCALE_PATTERN_HPP

#include <popart/patterns/patterns.hpp>
#include <popart/patterns/sequenceexpander.hpp>

namespace popart {

class NegativeOneScalePattern : public SequenceExpander {
public:
  bool matches(Op *) const override;

private:
  std::vector<std::unique_ptr<Op>> sequence(Op *op) const final;
};

} // namespace popart

#endif
