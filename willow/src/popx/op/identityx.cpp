#include <poponnx/error.hpp>
#include <poponnx/op/identity.hpp>
#include <poponnx/popx/op/identityx.hpp>

namespace poponnx {
namespace popx {

IdentityOpx::IdentityOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (!op->isConvertibleTo<IdentityOp>()) {
    throw error("cannot create IdentityOpx from " + op->op_type());
  }
}

void IdentityOpx::grow(poplar::program::Sequence &prog) const {
  insert(outId(0), Opx::cloneNcopy(prog, inId(0)));
}

IdentityGradOpx::IdentityGradOpx(Op *op, Devicex *devicex)
    : IdentityOpx(op, devicex) {
  if (!op->isConvertibleTo<IdentityGradOp>()) {
    throw error("cannot create IdentityGradOpx from " + op->op_type());
  }
}

} // namespace popx
} // namespace poponnx
