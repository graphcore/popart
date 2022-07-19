// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_NEGATIVEONESCALEPATTERN_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_NEGATIVEONESCALEPATTERN_HPP_

#include <memory>
#include <vector>
#include <popart/patterns/sequenceexpander.hpp>

namespace popart {
class Op;

class NegativeOneScalePattern : public SequenceExpander {
public:
  bool matches(Op *) const override;

private:
  std::vector<std::unique_ptr<Op>> sequence(Op *op) const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_NEGATIVEONESCALEPATTERN_HPP_
