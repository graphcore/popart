#ifndef GUARD_NEURALNET_STEPIO_HPP
#define GUARD_NEURALNET_STEPIO_HPP

#include <poponnx/error.hpp>
#include <poponnx/names.hpp>
#include <poponnx/tensorinfo.hpp>

#include <functional>
#include <numeric>
#include <ostream>

// TODO T5992:
// Consider merging NDIndices functionality into class Array,
// move Array out of this header, consider also the template class
// NDArray in ces/addce.cpp

namespace poponnx {

// A class to hold data, used
// within the poponnx::Tensor class.
class TensorData {
public:
  // create by copying from src to data_,
  // the size of the copy determined by TensorInfo
  TensorData(const TensorInfo &, const void *src);

  // create by copying to data_ from onnx::TensorProto
  TensorData(const onnx::TensorProto &);
  void *data();

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

// TODO : Rename Array to ArrayWrapper and then have template versions for the
// type information. See also T5992

class Array {
  std::vector<unsigned> shape;

public:
  Array(std::vector<unsigned> _shape) : shape(_shape) {}

  virtual ~Array() {}

  virtual void *getPtr()      = 0;
  virtual DataType getDtype() = 0;

  std::size_t getNdim() const { return shape.size(); }
  unsigned getShape(unsigned index) const { return shape[index]; }
  std::size_t numElements() const {
    return std::accumulate(
        shape.begin(), shape.end(), 1, std::multiplies<int>());
  }
};

template <typename TYPE> class ArrayWrapper : public Array {

  TYPE *data;

public:
  template <class T>
  friend std::ostream &operator<<(std::ostream &os,
                                  const ArrayWrapper<T> &array);

  // TODO : Fix it so ArrayWrapper can take a poponnx::Shape
  ArrayWrapper(std::vector<unsigned> _shape, TYPE *_data)
      : Array(_shape), data(_data) {}

  virtual void *getPtr() { return static_cast<void *>(data); }
  virtual DataType getDtype();
};

template <> DataType ArrayWrapper<float>::getDtype();

template <typename T>
std::ostream &operator<<(std::ostream &os, const ArrayWrapper<T> &array);

class StepIO : public IStepIO {
public:
  StepIO(std::map<TensorId, Array &> inputs_,
         std::map<TensorId, Array &> outputs_)
      : inputs(inputs_), outputs(outputs_) {}

  TensorInfo getTensorInfo(Array &array) const {
    auto dtype = array.getDtype();
    auto tRank = array.getNdim();
    std::vector<int64_t> shape;
    for (int i = 0; i < tRank; ++i) {
      shape.push_back(array.getShape(i));
    }
    return TensorInfo(dtype, shape);
  }

  template <typename T>
  T get(TensorId id,
        const std::map<TensorId, Array &> &M,
        std::string mapName) const {
    auto found = M.find(id);
    if (found == M.end()) {
      throw error("No tensor {} provided in CppStepIO's {}", id, mapName);
    }
    Array &npArr = found->second;
    T stepData;
    stepData.data = npArr.getPtr();
    stepData.info = getTensorInfo(npArr);
    return stepData;
  }

  ConstVoidData in(TensorId id) const final {
    return get<ConstVoidData>(id, inputs, "inputs");
  }

  MutableVoidData out(TensorId id) const final {
    return get<MutableVoidData>(id, outputs, "outputs");
  }

private:
  std::map<TensorId, Array &> inputs;
  std::map<TensorId, Array &> outputs;
};

} // namespace poponnx

#endif
