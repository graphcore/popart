#ifndef GUARD_NEURALNET_STEPIO_HPP
#define GUARD_NEURALNET_STEPIO_HPP

#include <willow/names.hpp>

namespace willow {

class StepInData {
public:
  const void *data;
  // This is used to confirm that data is as expected
  TensorInfo info;
};

class StepOutData {
public:
  void *data;
  // This is used to confirm that data is as expected
  TensorInfo info;
};

class StepIO {
public:
  virtual ~StepIO()                       = default;
  virtual StepInData in(TensorId) const   = 0;
  virtual StepOutData out(TensorId) const = 0;
};

} // namespace willow

#endif
