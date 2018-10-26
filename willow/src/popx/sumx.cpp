#include <willow/error.hpp>
#include <willow/popx/sumx.hpp>
#include <willow/sum.hpp>

namespace willow {
namespace popx {

SumOpx::SumOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::SUM) {
    throw error("cannot create SumOpx from " + op->op_type());
  }
}

SumOp *SumOpx::getSumOp() const { return dynamic_cast<SumOp *>(op_p); }

} // namespace popx
} // namespace willow
