// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_STEPIO_HPP
#define GUARD_NEURALNET_STEPIO_HPP
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <popart/iarray.hpp>
#include <popart/stepio_generic.hpp>

#include "popart/datatype.hpp"
#include "popart/istepio.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/voiddata.hpp"

namespace popart {

namespace popx {
class Executablex;
}

/**
 * @brief Class that implements the IStepIO interface using user-provided
 * callback functions.
 *
 * The IStepIO interface contains a number of pure virtual member functions
 * through which PopART receives buffers to read data from and buffers to write
 * data to.
 * This class inherits from IStepIO and implements those member functions
 * by delegating the logic to the callback functions passed in the constructor.
 * This gives the user full control as to how data buffers are provisioned.
 *
 * See IStepIO for more details on the expected behaviour of the callbacks.
 */
class StepIOCallback : public IStepIO {

public:
  /**
   * Callable object that implements IStepIO::in().
   */
  using InputCallback = std::function<ConstVoidData(TensorId, bool)>;
  /**
   * Callable object that implements IStepIO::inComplete().
   */
  using InputCompleteCallback = std::function<void(TensorId)>;
  /**
   * Callable object that implements IStepIO::out().
   */
  using OutputCallback = std::function<MutableVoidData(TensorId)>;
  /**
   * Callable object that implements IStepIO::outComplete().
   */
  using OutputCompleteCallback = std::function<void(TensorId)>;

  /**
   * @brief Construct a new StepIOCallback object.
   *
   * @param inputCallback The callback function the constructed
   *   StepIOCallback instance will use when IStepIO::in() is called.
   *   See IStepIO for details on how to implement this method.
   * @param inputCompleteCallback The callback function the constructed
   *   StepIOCallback instance will use when IStepIO::inComplete() is
   *   called. See IStepIO for details on how to implement this method.
   * @param outputCallback The callback function the constructed
   *   StepIOCallback instance will use when IStepIO::out() is called.
   *   See IStepIO for details on how to implement this method.
   * @param outputCompleteCallback The callback function the constructed
   *   StepIOCallback instance will use when IStepIO::outComplete() is
   * called. See IStepIO for details on how to implement this method.
   */
  StepIOCallback(InputCallback inputCallback,
                 InputCompleteCallback inputCompleteCallback,
                 OutputCallback outputCallback,
                 OutputCompleteCallback outputCompleteCallback)
      : inputCb(inputCallback), inputCompleteCb(inputCompleteCallback),
        outputCb(outputCallback), outputCompleteCb(outputCompleteCallback) {}

  void assertNumElements(const popx::Executablex &) const {}

  /**
   * This function is called by PopART when a StepIOCallback instance is
   * passed to Session::run() and will internally call the
   * `inputCallback` parameter passed to the constructor.
   *
   * You should not call this function directly.
   */
  ConstVoidData in(TensorId id, int64_t numElements, bool prefetch)final;

  /**
   * This function is called by PopART when a StepIOCallback instance is
   * passed to Session::run() and will internally call the
   * `inputCompleteCallback` parameter passed to the constructor.
   *
   * You should not call this function directly.
   */
  void inComplete(TensorId id, int64_t numElements) final;

  /**
   * This function is called by PopART when a StepIOCallback instance is
   * passed to Session::run() and will internally call the
   * `outputCallback` parameter passed to the constructor.
   *
   * You should not call this function directly.
   */
  MutableVoidData out(TensorId id, int64_t numElements) final;

  /**
   * This function is called by PopART when a StepIOCallback instance is
   * passed to Session::run() and will internally call the
   * `outputCompleteCallback` parameter passed to the constructor.
   *
   * You should not call this function directly.
   */
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
