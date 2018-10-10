#ifndef GUARD_NEURALNET_OPTIMIZER_HPP
#define GUARD_NEURALNET_OPTIMIZER_HPP

#include <willow/names.hpp>

namespace willow {

class Optimizer {
public:
  virtual ~Optimizer()                             = default;
  Optimizer()                                      = default;
  Optimizer(const Optimizer &)                     = default;
  virtual std::unique_ptr<Optimizer> clone() const = 0;
};

class SGD : public Optimizer {
public:
  SGD(float lr);
  float learnRate();
  virtual std::unique_ptr<Optimizer> clone() const override final;

private:
  float learnRate_;
};

} // namespace willow

#endif
