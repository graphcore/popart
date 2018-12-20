#include <poponnx/error.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/popx/op/identityx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

IdentityOpx::IdentityOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<IdentityOp>(op, Onnx::Operators::Identity);
}

void IdentityOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0), Opx::cloneNcopy(prog, inId(0)));
}

IdentityGradOpx::IdentityGradOpx(Op *op, Devicex *devicex)
    : IdentityOpx(op, devicex) {
  verifyOp<IdentityGradOp>(op, Onnx::GradOperators::IdentityGrad);
}

namespace {
OpxCreator<IdentityOpx> identityOpxCreator(Onnx::Operators::Identity);
OpxCreator<IdentityGradOpx>
    identityGradOpxCreator(Onnx::GradOperators::IdentityGrad);
} // namespace

} // namespace popx
} // namespace poponnx
