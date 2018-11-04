#ifndef GUARD_NEURALNET_STEPIO_HPP
#define GUARD_NEURALNET_STEPIO_HPP

#include <willow/names.hpp>
#include <willow/tensorinfo.hpp>

namespace willow {

class ConstVoidData {
public:
  const void *data;
  // This is used to confirm that data is as expected
  TensorInfo info;
};

class MutableVoidData {
public:
  void *data;
  // This is used to confirm that data is as expected
  TensorInfo info;
};

class StepIO {
public:
  virtual ~StepIO()                           = default;
  virtual ConstVoidData in(TensorId) const    = 0;
  virtual MutableVoidData out(TensorId) const = 0;
};

} // namespace willow

#endif
