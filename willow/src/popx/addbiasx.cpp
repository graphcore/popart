#include <poponnx/addbias.hpp>
#include <poponnx/error.hpp>
#include <poponnx/popx/addbiasx.hpp>

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <poplin/Convolution.hpp>
#include <popops/Reduce.hpp>
#pragma clang diagnostic pop // stop ignoring warnings

namespace willow {
namespace popx {

AddBiasOpx::AddBiasOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  if (op->opType != OpType::ADDBIAS) {
    throw error("cannot create AddBiasOpx from " + op->op_type());
  }
}

AddBiasDataGradOpx::AddBiasDataGradOpx(Op *op, Devicex *devicex)
    : IdentityOpx(op, devicex) {
  if (op->opType != OpType::ADDBIASDATAGRAD) {
    throw error("cannot create AddBiasDataGradOpx from " + op->op_type());
  }
}

void AddBiasOpx::grow() const {
  // Clone & copy the input tensor because poplin::addBias is in-place.
  const auto result = Opx::cloneNcopy(inId(AddBiasOp::dataInIndex()));
  poplin::addBias(
      graph(), result, get(inId(AddBiasOp::biasInIndex())), step(), idStr());
  insert(outId(0), result);
}

AddBiasOp *AddBiasOpx::getAddBiasOp() const {
  return dynamic_cast<AddBiasOp *>(op_p);
}

AddBiasBiasGradOpx::AddBiasBiasGradOpx(Op *op, Devicex *devicex)
    : ReduceSumOpx(op, devicex) {
  if (op_p->opType != OpType::ADDBIASBIASGRAD) {
    throw error("cannot create AddBiasBiasGradOpx from " + op_p->op_type());
  }
}

} // namespace popx
} // namespace willow
