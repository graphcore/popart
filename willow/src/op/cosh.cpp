#include <memory>
#include <popart/op/cosh.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

CoshOp::CoshOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> CoshOp::clone() const {
  return std::make_unique<CoshOp>(*this);
}

std::vector<std::unique_ptr<Op>> CoshOp::getGradOps() {
  throw error("CoshOp should be removed by pattern 'CoshOp' before call to "
              "CoshOp::getGradOps");
}

void CoshOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

namespace {
static OpCreator<CoshOp> coshOpCreator(Onnx::Operators::Cosh_9);
}

} // namespace popart
