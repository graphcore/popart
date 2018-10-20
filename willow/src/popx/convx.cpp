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

ConvDataGradOpx::ConvDataGradOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::CONVDATAGRAD) {
    throw error("cannot create ConvDataGradOpx from " + op->op_type());
  }
}

ConvDataGradOp *ConvDataGradOpx::getConvDataGradOp() const {
  return dynamic_cast<ConvDataGradOp *>(getOp());
}

ConvWeightsGradOpx::ConvWeightsGradOpx(Op *op) : Opx(op) {
  if (op->opType != OpType::CONVWEIGHTSGRAD) {
    throw error("cannot create ConvWeightsGradOpx from " + op->op_type());
  }
}

ConvWeightsGradOp *ConvWeightsGradOpx::getConvWeightsGradOp() const {
  return dynamic_cast<ConvWeightsGradOp *>(getOp());
}

} // namespace popx
} // namespace willow
