// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/optimizervalue.hpp>

#include <boost/functional/hash.hpp>

namespace popart {
void OptimizerValue::validReplacement(const OptimizerValue &rhs) const {
  if (isConst() != rhs.isConst()) {
    throw error("Can not replace a constant value with a non constant value.");
  }
  if (isConst() && (val() - rhs.val() != 0.0f)) {
    throw error("Values are constant but do not match.");
  }
}

OptimizerValue &OptimizerValue::operator=(const OptimizerValue &rhs) {
  val_     = rhs.val_;
  isConst_ = rhs.isConst_;
  return *this;
}
} // namespace popart

namespace std {
std::size_t std::hash<popart::OptimizerValue>::operator()(
    const popart::OptimizerValue &value) const {
  std::size_t seed = 0;
  boost::hash_combine(seed, value.val());
  boost::hash_combine(seed, value.isConst());
  return seed;
}
} // namespace std
