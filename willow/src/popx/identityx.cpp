#include <poponnx/error.hpp>
#include <poponnx/identity.hpp>
#include <poponnx/popx/identityx.hpp>

namespace willow {
namespace popx {

IdentityOpx::IdentityOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (dynamic_cast<IdentityOp *>(op) == nullptr) {
    throw error("cannot create IdentityOpx from " + op->op_type());
  }
}

void IdentityOpx::grow() const { insert(outId(0), Opx::cloneNcopy(inId(0))); }

IdentityGradOpx::IdentityGradOpx(Op *op, Devicex *devicex)
    : IdentityOpx(op, devicex) {
  if (dynamic_cast<IdentityGradOp *>(op) == nullptr) {
    throw error("cannot create IdentityGradOpx from " + op->op_type());
  }
}

} // namespace popx
} // namespace willow
