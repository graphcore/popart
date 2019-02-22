#include <onnx/onnx_pb.h>
#include <poponnx/ces/unsqueezece.hpp>
#include <poponnx/op.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ConstExprUnsqueeze::ConstExprUnsqueeze(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprUnsqueeze::compute() {
  char *data  = reinterpret_cast<char *>(inTensor(0)->tensorData()->data());
  auto nbytes = outInfo0().nbytes();
  return std::vector<char>(data, data + nbytes);
}

} // namespace poponnx
