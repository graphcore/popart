#include <memory>
#include <poponnx/op/tan.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

TanOp::TanOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseUnaryOp(_opid, settings_) {}

std::unique_ptr<Op> TanOp::clone() const {
  return std::make_unique<TanOp>(*this);
}

std::vector<std::unique_ptr<Op>> TanOp::getGradOps() {
  throw error("TanOp should be removed by pattern 'TanOp' before call to "
              "TanOp::getGradOps");
}

namespace {
static OpCreator<TanOp> tanOpCreator(Onnx::Operators::Tan_7);
}

} // namespace poponnx
