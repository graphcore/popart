// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OPTIMIZERVALUE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OPTIMIZERVALUE_HPP_

#include <cstddef>
#include <functional>
#include <utility>

namespace popart {

/**
 * A class used to represent values of hyper parameters.
 */
class OptimizerValue final {
public:
  /// Equivalent to OptimizerValue(0, false).
  OptimizerValue() = default;
  /// Equivalent to OptimizerValue(v, true).
  OptimizerValue(float v) : val_(v), isConst_(true) {}
  /// Constructor.
  /// \param v The current value of the hyper parameter.
  /// \param c A boolean flag to indicate whether the parameter will remain
  ///     at this value forever (`true`) or may change over time (`false`).
  OptimizerValue(float v, bool c) : val_(v), isConst_(c) {}
  OptimizerValue(std::pair<float, bool> x)
      : OptimizerValue(x.first, x.second) {}

  // current value
  float val() const { return val_; }

  // can the user not change this value in the final computation Graph
  bool isConst() const { return isConst_; }

  void validReplacement(const OptimizerValue &rhs) const;

  bool operator==(const OptimizerValue &rhs) const;

private:
  float val_;
  bool isConst_;
};
} // namespace popart

namespace std {
template <> struct hash<popart::OptimizerValue> {
  std::size_t operator()(const popart::OptimizerValue &value) const;
};
} // namespace std

namespace popart {
inline std::size_t hash_value(const OptimizerValue &value) {
  return std::hash<OptimizerValue>()(value);
}
} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OPTIMIZERVALUE_HPP_
