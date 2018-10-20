#include <willow/error.hpp>
#include <willow/popx/sumx.hpp>
#include <willow/sum.hpp>

namespace willow {
namespace popx {

SumOpx::SumOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::SUM) {
    throw error("cannot create SumOpx from " + op->op_type());
  }
}

SumOp *SumOpx::getSumOp() const { return dynamic_cast<SumOp *>(getOp()); }

} // namespace popx
} // namespace willow
