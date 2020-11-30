// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/optimizervalue.hpp>

#include <boost/functional/hash.hpp>

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

namespace std {
std::size_t std::hash<popart::OptimizerValue>::operator()(
    const popart::OptimizerValue &value) const {
  std::size_t seed = 0;
  boost::hash_combine(seed, value.val());
  boost::hash_combine(seed, value.isConst());
  return seed;
}
} // namespace std
