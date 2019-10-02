#include <onnx/onnx_pb.h>
#include <popart/ces/identityce.hpp>
#include <popart/op.hpp>
#include <popart/tensor.hpp>

namespace popart {

ConstExprIdentity::ConstExprIdentity(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprIdentity::compute() {
  if (inTensor(0)->info.nbytes() != outInfo0().nbytes()) {
    throw error("This is not what identity should be doing");
  }
  char *data  = reinterpret_cast<char *>(inTensor(0)->tensorData()->data());
  auto nbytes = outInfo0().nbytes();
  return std::vector<char>(data, data + nbytes);
}

} // namespace popart
