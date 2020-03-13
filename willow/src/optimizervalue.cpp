// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/optimizervalue.hpp>

namespace popart {
bool OptimizerValue::validReplacement(const OptimizerValue &rhs) const {
  if (isConst() != rhs.isConst()) {
    return false;
  }
  if (isConst() && (val() - rhs.val() != 0.0f)) {
    return false;
  }
  return true;
}

OptimizerValue &OptimizerValue::operator=(const OptimizerValue &rhs) {
  val_     = rhs.val_;
  isConst_ = rhs.isConst_;
  return *this;
}
} // namespace popart
