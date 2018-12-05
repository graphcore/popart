#include <poponnx/makeunique.hpp>
#include <poponnx/op/tan.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

TanOp::TanOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::unique_ptr<Op> TanOp::clone() const { return make_unique<TanOp>(*this); }

std::vector<std::unique_ptr<Op>> TanOp::getGradOps() {
  throw error("TanOp should be removed by pattern 'TanOp' before call to "
              "TanOp::getGradOps");
}

void TanOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

} // namespace poponnx
