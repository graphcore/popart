#include <poponnx/makeunique.hpp>
#include <poponnx/op/cosh.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

CoshOp::CoshOp(const OpConstructorBundle &bundle) : Op(bundle) {}

CoshOp::CoshOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {}

std::unique_ptr<Op> CoshOp::clone() const { return make_unique<CoshOp>(*this); }

std::vector<std::unique_ptr<Op>> CoshOp::getGradOps() {
  throw error("CoshOp should be removed by pattern 'CoshOp' before call to "
              "CoshOp::getGradOps");
}

void CoshOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

} // namespace poponnx
