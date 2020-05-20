// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DEVICEREDUCE_HPP
#define GUARD_NEURALNET_DEVICEREDUCE_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

// Transform to insert replicated reduction operations before VarUpdates for
// data parallel training
class DeviceReduce : public Transform {
public:
  static std::size_t id();

  DeviceReduce() : Transform() {}

  virtual bool apply(Graph &) const final;

  virtual std::size_t getId() const final { return id(); }

  virtual std::string getName() const final { return "DeviceReduce"; }

private:
  // Generate a name for the new reduced tensor
  TensorId generateReducedTensorId(Tensor *tensor) const;
};

} // namespace popart

#endif
