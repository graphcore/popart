#include <onnx/onnx_pb.h>
#include <popart/ces/unsqueezece.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>

namespace popart {

ConstExprUnsqueeze::ConstExprUnsqueeze(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprUnsqueeze::compute() {
  char *data  = reinterpret_cast<char *>(inTensor(0)->tensorData()->data());
  auto nbytes = outInfo0().nbytes();
  return std::vector<char>(data, data + nbytes);
}

} // namespace popart
