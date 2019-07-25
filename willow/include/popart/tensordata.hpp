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
  const void *data;
  // This is used to confirm that data is as expected
  TensorInfo info;
};

// A class to point to non-const data
class MutableVoidData {
public:
  void *data;
  // This is used to confirm that data is as expected
  TensorInfo info;
};

// A virtual class for accessing pointers to
// the data required to perform a training step
class IStepIO {
public:
  virtual ~IStepIO() = default;
  // constant input data,
  virtual ConstVoidData in(TensorId) const = 0;
  // non-const anchor data,
  // which will be modified inplace.
  virtual MutableVoidData out(TensorId) const = 0;
};

class StepIO : public IStepIO {
public:
  StepIO(std::map<TensorId, IArray &> inputs_,
         std::map<TensorId, IArray &> outputs_)
      : inputs(inputs_), outputs(outputs_) {}

  ConstVoidData in(TensorId id) const final;
  MutableVoidData out(TensorId id) const final;

private:
  TensorInfo getTensorInfo(IArray &array) const;

  template <typename T>
  T get(TensorId id,
        const std::map<TensorId, IArray &> &M,
        std::string mapName) const;

  std::map<TensorId, IArray &> inputs;
  std::map<TensorId, IArray &> outputs;
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
