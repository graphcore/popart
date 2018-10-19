#include <willow/conv.hpp>
#include <willow/error.hpp>
#include <willow/popx/opx.hpp>

namespace willow {
namespace popx {

Opx::Opx(Op *op_) : op(op_) {}
Opx::~Opx() = default;

poplar::Tensor Opx::createInput(int) const {
  throw error("Opx for " + op->op_type() + " cannot create Input");
}

Op *Opx::getOp() const { return op; }

bool Opx::canCreateInput(int) const { return false; }

} // namespace popx
} // namespace willow
