#include <vector>
#include <poponnx/ces/shapece.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

void ConstExprShape::insertOutput() {
  std::vector<char> data;
  Tensor *input = atInIndex(0);
  auto &shape   = input->info.shape();

  data.resize(shape.size() * sizeof(int64_t));
  int64_t *int_data = reinterpret_cast<int64_t *>(data.data());

  for (int i = 0; i < shape.size(); i++) {
    int_data[i] = shape[i];
  }

  TensorInfo outInfo(DataType::INT64, {static_cast<int64_t>(shape.size())});
  addConstInitTensor(atOutIndex0(), outInfo, data.data());
}

} // namespace poponnx
