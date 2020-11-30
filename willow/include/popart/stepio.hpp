// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STEPIO_HPP
#define GUARD_NEURALNET_STEPIO_HPP
#include <popart/iarray.hpp>
#include <popart/stepio_generic.hpp>
#include <popart/tensorinfo.hpp>

#include <functional>
#include <iostream>
#include <numeric>
#include <ostream>

namespace popart {

namespace popx {
class Executablex;
}

class StepIOCallback : public IStepIO {

public:
  using InputCallback          = std::function<ConstVoidData(TensorId, bool)>;
  using InputCompleteCallback  = std::function<void(TensorId)>;
  using OutputCallback         = std::function<MutableVoidData(TensorId)>;
  using OutputCompleteCallback = std::function<void(TensorId)>;

  StepIOCallback(InputCallback inputCb_,
                 InputCompleteCallback inputCompleteCb_,
                 OutputCallback outputCb_,
                 OutputCompleteCallback outputCompleteCb_)
      : inputCb(inputCb_), inputCompleteCb(inputCompleteCb_),
        outputCb(outputCb_), outputCompleteCb(outputCompleteCb_) {}

  void assertNumElements(const popx::Executablex &) const {}

  ConstVoidData in(TensorId id, int64_t numElements, bool prefetch)final;
  void inComplete(TensorId id, int64_t numElements) final;

  MutableVoidData out(TensorId id, int64_t numElements) final;
  void outComplete(TensorId id) final;

private:
  InputCallback inputCb;
  InputCompleteCallback inputCompleteCb;
  OutputCallback outputCb;
  OutputCompleteCallback outputCompleteCb;
};

// A virtual class for accessing pointers to
// the data required to perform a training step
class IWeightsIO {
public:
  virtual ~IWeightsIO() = default;

  virtual bool contains(TensorId) const = 0;

  virtual MutableVoidData weight(TensorId) const = 0;
};

class WeightsIO : public IWeightsIO {
public:
  virtual ~WeightsIO() override = default;
  virtual bool contains(TensorId) const final;
  virtual MutableVoidData weight(TensorId) const final;
  void insert(TensorId, MutableVoidData);

private:
  std::map<TensorId, MutableVoidData> weights;
};

namespace StepIONS {

struct IArrayAccessor {

  static void *getDataPointer(IArray &array) { return array.data(); }

  static size_t getArraySize(const IArray &array) { return array.nelms(); }

  static DataType getArrayDataType(IArray &array) { return array.dataType(); }

  static size_t getArrayRank(IArray &array) { return array.rank(); }

  static int64_t getArrayDim(IArray &array, size_t index) {
    return array.dim(index);
  }
};
} // namespace StepIONS
class StepIO
    : public StepIOGeneric<IArray, StepIONS::IArrayAccessor, IArray &> {
public:
  StepIO(std::map<TensorId, IArray &> inputs,
         std::map<TensorId, IArray &> outputs) {

    for (auto p : inputs) {
      inputsInfo.insert({p.first, {p.second, 0}});
    }

    for (auto p : outputs) {
      outputsInfo.insert({p.first, {p.second, 0}});
    }
  }
};

} // namespace popart

#endif
