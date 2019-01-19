#include <vector>
#include <poponnx/ces/transposece.hpp>
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

// transpose a tensor
template <typename T>
std::vector<char> transpose(Tensor *in0, const Shape &perm) {

  Shape shape;
  for (auto d : perm) {
    shape.push_back(in0->info.shape()[d]);
  }

  TensorInfo outInfo(in0->info.data_type(), shape);
  std::vector<char> v_out(outInfo.nbytes());
  NDArray<T> output(reinterpret_cast<T *>(v_out.data()), outInfo);

  NDArray<T> data0(static_cast<T *>(in0->tensorData()->data()), in0->info);

  for (int64_t i = 0; i < outInfo.nelms(); ++i) {
    // the N-dimensional indices in the output tensor
    auto indices = data0.ndindices.unflatten(i);

    // re-arrange the indices according to perm
    Shape pindices;
    for (auto d : perm) {
      pindices.push_back(indices[d]);
    }

    // Move the value
    output.at(i) = data0.at(output.ndindices.flatten(pindices));
  }

  return v_out;
}

void ConstExprTranspose::insertOutput() {
  std::vector<char> data_;
  Tensor *in0 = atInIndex(0);

  Shape perm;
  nAtts.setIfPresent(perm, "perm");

  if (perm.empty()) {
    // Default is to reverse the input shape
    for (int64_t i = in0->info.rank() - 1; i >= 0; i--) {
      perm.push_back(i);
    }
  }

  // verify that perm is a valid permutation
  std::vector<int> present(in0->info.rank(), 0);
  for (auto &x : perm) {
    if (x >= 0 && x < in0->info.rank()) {
      present[x] = 1;
    }
  }
  if (!std::accumulate(
          present.begin(), present.end(), 1, std::multiplies<int>())) {
    throw error("invalid permutation in ConstExprTranspose");
  }

  // Determine the output shape
  Shape outShape;
  for (auto d : perm) {
    outShape.push_back(in0->info.shape()[d]);
  }

  TensorInfo outInfo(in0->info.data_type(), outShape);
  if (in0->info.dataType() == DataType::INT64) {
    data_ = transpose<int64_t>(in0, perm);
  } else if (in0->info.dataType() == DataType::INT32) {
    data_ = transpose<int>(in0, perm);
  } else if (in0->info.dataType() == DataType::FLOAT) {
    data_ = transpose<float>(in0, perm);
  } else if (in0->info.dataType() == DataType::FLOAT16) {
    // any type which is 16-bits will suffice here, using uint16_t
    data_ = transpose<uint16_t>(in0, perm);
  } else {
    throw error("Currently ConstExprTranspose does not support type {}",
                in0->info.data_type());
  }

  addConstInitTensor(atOutIndex0(), outInfo, data_.data());
}

} // namespace poponnx
