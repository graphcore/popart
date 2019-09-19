
#ifndef GUARD_NEURALNET_OPTIMIZERVALUE_HPP
#define GUARD_NEURALNET_OPTIMIZERVALUE_HPP

namespace popart {

class OptimizerValue {
public:
  OptimizerValue(float v) : val_(v), isConst_(true) {}
  OptimizerValue(float v, bool c) : val_(v), isConst_(c) {}
  OptimizerValue(const OptimizerValue &) = default;
  OptimizerValue()                       = delete;
  ~OptimizerValue()                      = default;

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
