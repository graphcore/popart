#ifndef GUARD_NEURALNET_OPTIMIZERVALUE_HPP
#define GUARD_NEURALNET_OPTIMIZERVALUE_HPP

#include <tuple>

namespace popart {

class OptimizerValue {
public:
  OptimizerValue() = default;
  OptimizerValue(float v) : val_(v), isConst_(true) {}
  OptimizerValue(float v, bool c) : val_(v), isConst_(c) {}
  OptimizerValue(std::pair<float, bool> x)
      : OptimizerValue(x.first, x.second) {}

  OptimizerValue(const OptimizerValue &) = default;
  ~OptimizerValue()                      = default;
  OptimizerValue &operator               =(const OptimizerValue &rhs);

  // current value
  float val() const { return val_; }

  // can the user not change this value in the final computation Graph
  bool isConst() const { return isConst_; }

  bool validReplacement(const OptimizerValue &rhs) const;

private:
  float val_;
  bool isConst_;
};

} // namespace popart

#endif
