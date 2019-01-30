#include <vector>
#include <poponnx/ces/addce.hpp>
#include <poponnx/ndindices.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

template <typename T> class NDArray {
public:
  NDArray(T *d, const TensorInfo &i) : data(d), info(i), ndindices(info) {}
  T &at(int64_t i) { return data[i]; }
  T &at(const std::vector<int64_t> &indices) {
    return at(ndindices.flatten(indices));
  }
  T *data;
  const TensorInfo &info;
  NDIndices ndindices;
};

// add two Tensors together using numpy-broadcasting,
// return the data as a vector<char>
template <typename T> std::vector<char> add(Tensor *in0, Tensor *in1) {
  TensorInfo outInfo = npOut(in0->info, in1->info);
  std::vector<char> v_out(outInfo.nbytes());
  NDArray<T> output(reinterpret_cast<T *>(v_out.data()), outInfo);
  NDArray<T> data0(static_cast<T *>(in0->tensorData()->data()), in0->info);
  NDArray<T> data1(static_cast<T *>(in1->tensorData()->data()), in1->info);
  for (int64_t i = 0; i < outInfo.nelms(); ++i) {
    // the N-dimensional indices in the output tensor
    auto indices = output.ndindices.unflatten(i);
    // perform the addition, where the broadcasting of the
    // operands is implicitly taken care of by NDIndices
    output.at(i) = data0.at(indices) + data1.at(indices);
  }
  return v_out;
}

void ConstExprAdd::insertOutput() {
  std::vector<char> data_;
  Tensor *in0        = atInIndex(0);
  Tensor *in1        = atInIndex(1);
  TensorInfo outInfo = npOut(in0->info, in1->info);
  if (in0->info.dataType() == DataType::INT64) {
    data_ = add<int64_t>(in0, in1);
  } else if (in0->info.dataType() == DataType::INT32) {
    data_ = add<int>(in0, in1);
  } else if (in0->info.dataType() == DataType::FLOAT) {
    data_ = add<float>(in0, in1);
  } else {
    throw error("Currently ConstExprAdd does not support type {}",
                in0->info.data_type());
  }
  addConstInitTensor(atOutIndex0(), outInfo, data_.data());
}

} // namespace poponnx
