#include <willow/conv.hpp>
#include <willow/error.hpp>
#include <willow/popx/opx.hpp>

namespace willow {
namespace popx {

Opx::Opx(Op *op__, Devicex *devicex__) : op_(op__), devicex_(devicex__) {}
Opx::~Opx() = default;

poplar::Tensor Opx::createInput(int) const {
  throw error("Opx for " + op_->op_type() + " cannot create Input");
}

bool Opx::createsEquiv(int, Opx *, int) const {
  throw error("No check for equivalent tensor create for type " +
              op_->op_type());
}

std::vector<TensorId> Opx::mustExistBeforeCreate(int index0) const {
  throw error(
      "Opx for " + op_->op_type() +
      " cannot say which poplar Tensors must exist to create at index " +
      std::to_string(index0));
}

Op *Opx::getOp() const { return op_; }

Devicex *Opx::getDevx() const { return devicex_; }

bool Opx::canCreateInput(int) const { return false; }

} // namespace popx
} // namespace willow
