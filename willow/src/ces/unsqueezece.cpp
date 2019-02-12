#include <onnx/onnx_pb.h>
#include <poponnx/ces/unsqueezece.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ConstExprUnsqueeze::ConstExprUnsqueeze(const onnx::NodeProto &n, Ir *i)
    : ConstExprOp(n, i) {}

void ConstExprUnsqueeze::insertOutput() {
  std::vector<int64_t> axes = nAtts.getAttribute<Attributes::Ints>("axes");
  auto in_info              = atInIndex(0)->info;
  auto out_shape            = unsqueeze(in_info.shape(), axes);

  TensorInfo out_info{in_info.dataType(), out_shape};

  addConstInitTensor(
      atOutIndex0(), out_info, atInIndex(0)->tensorData()->data());
}

} // namespace poponnx
