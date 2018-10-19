#include <willow/conv.hpp>
#include <willow/error.hpp>
#include <willow/popx/convx.hpp>

namespace willow {
namespace popx {

ConvOpx::ConvOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::CONV) {
    throw error("cannot create ConvOpx from " + op->op_type());
  }
}

ConvOp *ConvOpx::getConvOp() const { return dynamic_cast<ConvOp *>(getOp()); }

bool ConvOpx::canCreateInput(int) const { return true; }

poplar::Tensor ConvOpx::createInput(int index) const {
  throw error("I know I said I could");
}

} // namespace popx
} // namespace willow
