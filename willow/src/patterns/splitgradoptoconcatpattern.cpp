// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/concat.hpp>
#include <popart/op/split.hpp>
#include <popart/patterns/splitgradoptoconcatpattern.hpp>

#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/region.hpp" // IWYU pragma: keep

namespace popart {

bool SplitGradOpToConcatPattern::matches(Op *op) const {
  return op->isConvertibleTo<SplitGradOp>();
}

std::vector<std::unique_ptr<Op>>
SplitGradOpToConcatPattern::sequence(Op *op) const {
  auto splitGradOp = dynamic_cast<SplitGradOp *>(op);

  std::vector<std::unique_ptr<Op>> seq;

  auto axis = splitGradOp->getAxis();
  seq.push_back(std::make_unique<ConcatOp>(
      Onnx::Operators::Concat_4, axis, op->getSettings()));

  return seq;
}

namespace {
static PatternCreator<SplitGradOpToConcatPattern>
    splitGradOpToConcatPattern("SplitGradOpToConcat");
}

} // namespace popart
