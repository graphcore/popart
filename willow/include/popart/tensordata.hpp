#ifndef GUARD_NEURALNET_STEPIO_HPP
#define GUARD_NEURALNET_STEPIO_HPP

#include <popart/error.hpp>
#include <popart/iarray.hpp>
#include <popart/names.hpp>
#include <popart/tensorinfo.hpp>

#include <functional>
#include <numeric>
#include <ostream>

namespace popart {

class Tensors;
class Ir;

// A class to hold data, used
// within the popart::Tensor class.
class TensorData {
public:
  // create by copying from src to data_,
  // the size of the copy determined by TensorInfo
  TensorData(const TensorInfo &, const void *src);

  // create by copying to data_ from onnx::TensorProto
  TensorData(const onnx::TensorProto &);

  void *data();
  const void *data() const;

  // reset the data in the TensorData by copying from src.
  // Input data must be the same size as the existing data_
  void resetData(const TensorInfo &, const void *src);

  // reset the data in the TensorData bt copying from onnx::TensorProto.
  // Input data must be the same size as the existing data_
  void resetData(const onnx::TensorProto &);

private:
  std::vector<char> data_;
};

// A class to point to constant data
class ConstVoidData {
public:
  ConstVoidData() = default;
  ConstVoidData(const void *data_, const TensorInfo &info_);

  const void *data = nullptr;
  // This is used to confirm that data is as expected
  TensorInfo info;

  bool storesData() const { return hasOptionalData; }
  void store(std::vector<char> &&d, const TensorInfo &i);

private:
  std::vector<char> optionalData;
  bool hasOptionalData{false};
};

// A class to point to non-const data
class MutableVoidData {
public:
  void *data = nullptr;
  // This is used to confirm that data is as expected
  TensorInfo info;
};

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

class StepIO : public IStepIO {

  struct ArrayInfo {
    IArray &array;
    int64_t offset;
  };

public:
  StepIO(std::map<TensorId, IArray &> inputs_,
         std::map<TensorId, IArray &> outputs_);

  ConstVoidData in(TensorId id, int64_t numElements, bool prefetch)final;
  void inComplete(TensorId id, int64_t numElements) final;
  MutableVoidData out(TensorId id, int64_t numElements) final;

  virtual void assertNumElements(const Ir &) const final;

private:
  TensorInfo getTensorInfo(IArray &array) const;

  template <typename T>
  T get(TensorId id,
        std::map<TensorId, ArrayInfo> &M,
        int64_t numElements,
        bool advance,
        std::string mapName);

  template <typename T>
  void advance(TensorId id,
               std::map<TensorId, ArrayInfo> &M,
               int64_t numElements,
               std::string mapName);

  std::map<TensorId, ArrayInfo> outputsInfo;
  std::map<TensorId, ArrayInfo> inputsInfo;
};

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

  void assertNumElements(const Ir &) const {}

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

} // namespace popart

#endif
