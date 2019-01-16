#include <poponnx/makeunique.hpp>
#include <poponnx/op/cosh.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

CoshOp::CoshOp(const OperatorIdentifier &_opid,
               Ir *_ir,
               const std::string &name,
               const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {}

std::unique_ptr<Op> CoshOp::clone() const { return make_unique<CoshOp>(*this); }

std::vector<std::unique_ptr<Op>> CoshOp::getGradOps() {
  throw error("CoshOp should be removed by pattern 'CoshOp' before call to "
              "CoshOp::getGradOps");
}

void CoshOp::setup() { outInfo(getOutIndex()) = inInfo(getInIndex()); }

namespace {
static OpCreator<CoshOp> coshOpCreator(Onnx::Operators::Cosh_9);
}

} // namespace poponnx
