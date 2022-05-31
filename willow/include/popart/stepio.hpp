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
 * Class that implements the IStepIO interface using user-provided
 * callback functions.
 *
 * The IStepIO interface contains a number of pure virtual member functions
 * through which PopART receives buffers to read data from and buffers to write
 * data to.
 * StepIOCallback inherits from IStepIO and implements those member functions
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
   * Construct a StepIOCallback object.
   *
   * \param inputCallback The callback function the constructed
   *      StepIOCallback instance will use when IStepIO::in() is called.
   *      See IStepIO for details on how to implement this method.
   * \param inputCompleteCallback The callback function the constructed
   *      StepIOCallback instance will use when IStepIO::inComplete() is
   *      called. See IStepIO for details on how to implement this method.
   * \param outputCallback The callback function the constructed
   *      StepIOCallback instance will use when IStepIO::out() is called.
   *      See IStepIO for details on how to implement this method.
   * \param outputCompleteCallback The callback function the constructed
   *      StepIOCallback instance will use when IStepIO::outComplete() is
   *      called. See IStepIO for details on how to implement this method.
   */
  StepIOCallback(InputCallback inputCallback,
                 InputCompleteCallback inputCompleteCallback,
                 OutputCallback outputCallback,
                 OutputCompleteCallback outputCompleteCallback)
      : inputCb(inputCallback), inputCompleteCb(inputCompleteCallback),
        outputCb(outputCallback), outputCompleteCb(outputCompleteCallback) {}

  /**
   * Check number of elements.
   *
   * This check is performed when IStepIO::runtimeAssertsEnabled() is `true`.
   *
   * \param Executablex The input executable to be checked that the input and
   *    output buffers have the correct number of elements.
   */
  void assertNumElements(const popx::Executablex &) const {}

  /**
   * This function is called by PopART when a StepIOCallback instance is
   * passed to Session::run() and will internally call the
   * \c inputCallback parameter passed to the constructor.
   *
   * This function should not be called directly.
   */
  ConstVoidData in(TensorId id, int64_t numElements, bool prefetch)final;

  /**
   * This function is called by PopART when a StepIOCallback instance is
   * passed to Session::run() and will internally call the
   * `inputCompleteCallback` parameter passed to the constructor.
   *
   * This function should not be called directly.
   */
  void inComplete(TensorId id, int64_t numElements) final;

  /**
   * This function is called by PopART when a StepIOCallback instance is
   * passed to Session::run() and will internally call the
   * `outputCallback` parameter passed to the constructor.
   *
   * This function should not be called directly.
   */
  MutableVoidData out(TensorId id, int64_t numElements) final;

  /**
   * This function is called by PopART when a StepIOCallback instance is
   * passed to Session::run() and will internally call the
   * `outputCompleteCallback` parameter passed to the constructor.
   *
   * This function should not be called directly.
   */
  void outComplete(TensorId id) final;

private:
  InputCallback inputCb;
  InputCompleteCallback inputCompleteCb;
  OutputCallback outputCb;
  OutputCompleteCallback outputCompleteCb;
};

/**
 * A virtual class for accessing pointers to
 * the data required to perform a training step.
 */
class IWeightsIO {
public:
  /// Destructor for IWeightsIO.
  virtual ~IWeightsIO() = default;

  /**
   * Check if the WeightsIO instance contains the weights for a specific tensor.
   *
   * \param TensorId The ID of the tensor to look for weights for.
   * \returns `true` if the WeightsIO instance contains weights for the tensor,
   *      `false` otherwise.
   */
  virtual bool contains(TensorId) const = 0;

  /**
   * Retrieve weights for a specific tensor.
   *
   * \param TensorId The ID of the tensor to retrieve weights for.
   * \returns The weights.
   */
  virtual MutableVoidData weight(TensorId) const = 0;
};

/// Class representing weights.
class WeightsIO : public IWeightsIO {
public:
  /// Destructor for WeightsIO.
  virtual ~WeightsIO() override = default;

  /**
   * Check if the WeightsIO instance contains the weights for a specific tensor.
   *
   * \param TensorId The ID of the tensor to look for weights for.
   * \returns `true` if the WeightsIO instance contains weights for the tensor,
   *      `false` otherwise.
   */
  virtual bool contains(TensorId) const final;

  /**
   * Retrieve weights for a specific tensor from the WeightsIO object.
   *
   * \param TensorId The ID of the tensor to retrieve weights for.
   * \returns The weights.
   */
  virtual MutableVoidData weight(TensorId) const final;

  /**
   * Insert weights for a specific tensor into the WeightsIO object.
   *
   * \param TensorId The ID of the tensor to insert weights for.
   * \param MutableVoidData The weights to insert.
   */
  void insert(TensorId, MutableVoidData);

private:
  std::map<TensorId, MutableVoidData> weights;
};

namespace StepIONS {

/// Structure to help with accessing the data in IArray objects.
struct IArrayAccessor {

  /**
   * Get pointer to the data.
   * \param array The IArray object.
   * \returns A pointer to the data contained in the IArray object.
   */
  static void *getDataPointer(IArray &array) { return array.data(); }

  /**
   * Get the number of data elements.
   * \param array The IArray object.
   * \returns The number of data elements.
   */
  static size_t getArraySize(const IArray &array) { return array.nelms(); }

  /**
   * Get the data type of the data.
   * \param array The IArray object.
   * \returns The data type of the data.
   */
  static DataType getArrayDataType(IArray &array) { return array.dataType(); }

  /**
   * Get the rank of the data array.
   * \param array The IArray object.
   * \returns The rank of the data array.
   */
  static size_t getArrayRank(IArray &array) { return array.rank(); }

  /**
   * Get the size of the data at a specific location.
   * \param array The IArray object.
   * \param index The index of the data element in the IArray object.
   * \returns The size of the data at the specific location.
   */
  static int64_t getArrayDim(IArray &array, size_t index) {
    return array.dim(index);
  }
};
} // namespace StepIONS

/// Class to provide a Session object with input and output data.
class StepIO
    : public StepIOGeneric<IArray, StepIONS::IArrayAccessor, IArray &> {
public:
  /**
   * Constructor for StepIO.
   *
   * \param inputs The input data.
   * \param outputs The output data.
   */
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
