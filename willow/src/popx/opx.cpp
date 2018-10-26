#include <willow/conv.hpp>
#include <willow/error.hpp>
#include <willow/popx/opx.hpp>

namespace willow {
namespace popx {

Opx::Opx(Op *op_p_, Devicex *dv_p_) : op_p(op_p_), dv_p(dv_p_) {}
Opx::~Opx() = default;

poplar::Tensor Opx::createInput(int) const {
  throw error("Opx for " + op_p->op_type() + " cannot create Input");
}

bool Opx::createsEquiv(int, Opx *, int) const {
  throw error("No check for equivalent tensor create for type " +
              op_p->op_type());
}

std::vector<TensorId> Opx::mustExistBeforeCreate(int index0) const {
  throw error(
      "Opx for " + op_p->op_type() +
      " cannot say which poplar Tensors must exist to create at index " +
      std::to_string(index0));
}

void Opx::grow() const{
  throw error("adding poplar::Tensors not implemented for " + op_p->op_type());
}


bool Opx::canCreateInput(int) const { return false; }

} // namespace popx
} // namespace willow
