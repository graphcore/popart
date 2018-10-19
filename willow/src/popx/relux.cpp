#include <willow/error.hpp>
#include <willow/popx/relux.hpp>
#include <willow/relu.hpp>

namespace willow {
namespace popx {

ReluOpx::ReluOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::RELU) {
    throw error("cannot create ReluOpx from " + op->op_type());
  }
}

ReluOp *ReluOpx::getReluOp() const { return dynamic_cast<ReluOp *>(getOp()); }

} // namespace popx
} // namespace willow
