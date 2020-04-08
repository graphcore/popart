// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_ISTEPIO_HPP
#define GUARD_NEURALNET_ISTEPIO_HPP

#include <popart/names.hpp>
#include <popart/voiddata.hpp>
namespace popart {

// A virtual class for accessing pointers to
// the data required to perform a training step
class IStepIO {
public:
  virtual ~IStepIO() = default;
  // constant input data,
  virtual ConstVoidData in(TensorId id, int64_t numElements, bool prefetch) = 0;
  virtual void inComplete(TensorId id, int64_t numElements)                 = 0;

  // non-const anchor data,
  // which will be modified inplace.
  virtual MutableVoidData out(TensorId id, int64_t numElements) = 0;

  // Use to indicate then the output data has been written to the
  // MutableVoidData
  virtual void outComplete(TensorId) {}

  void enableRuntimeAsserts(bool b) { runtimeAssertsOn = b; }
  bool runtimeAssertsEnabled() const { return runtimeAssertsOn; }
  virtual void assertNumElements(const Ir &) const = 0;

private:
  bool runtimeAssertsOn{true};
};
} // namespace popart

#endif
