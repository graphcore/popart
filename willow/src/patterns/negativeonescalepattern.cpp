// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/math/special_functions/relative_difference.hpp>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/scale.hpp>
#include <popart/patterns/negativeonescalepattern.hpp>

#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"

using boost::math::epsilon_difference;

namespace popart {

bool NegativeOneScalePattern::matches(Op *op) const {
  if (!(op->isConvertibleTo<ScaleOp>())) {
    return false;
  }

  float scale_factor = dynamic_cast<ScaleOp *>(op)->getScaleFactor();
  return epsilon_difference(scale_factor, -1.0f) < 1.0f;
}

// output = neg(x)
std::vector<std::unique_ptr<Op>>
NegativeOneScalePattern::sequence(Op *op) const {
  std::vector<std::unique_ptr<Op>> seq;

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Neg, op));

  return seq;
}

namespace {
static PatternCreator<NegativeOneScalePattern>
    negativeOneScalePattern("NegativeOneScale");
}

} // namespace popart
