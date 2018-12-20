#include <poponnx/makeunique.hpp>
#include <poponnx/op/tan.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

TanOp::TanOp(const OperatorIdentifier &_opid,
             Ir *_ir,
             const std::string &name,
             const Attributes &_attr)
    : ElementWiseUnaryOp(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> TanOp::clone() const { return make_unique<TanOp>(*this); }

std::vector<std::unique_ptr<Op>> TanOp::getGradOps() {
  throw error("TanOp should be removed by pattern 'TanOp' before call to "
              "TanOp::getGradOps");
}

namespace {
static OpCreator<TanOp> tanOpCreator(Onnx::Operators::Tan);
}

} // namespace poponnx
