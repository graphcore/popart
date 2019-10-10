#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/varupdatex.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Collectives.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

VarUpdateOpx::VarUpdateOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}



SGDVarUpdateOpx::SGDVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<SGDVarUpdateOp>(op, Onnx::CustomOperators::SgdVarUpdate);
}

void SGDVarUpdateOpx::grow(poplar::program::Sequence &prog) const {

  // Weight update (matching pytorch implementation)
  //  w <- w * (1 - lr * wd) - (lr/ls) * weight_gradient
  //
  // lr = learning rate
  // ls = loss scaling
  // wd = weight decay
  //
  // This is expressed as
  //
  // w <- w * weightDecayScaleFactor - scaledLearningRate * weight_gradient
  //
  // The (1 - lr * wd) and (lr/ls) calculations are done in SGD::setTensorData

  auto vu_op = getOp<SGDVarUpdateOp>();

  // (1) update weights with weight decay

  // non-const weight decay scale factor
  if (!vu_op.initWeightDecayScaleFactor.isConst()) {

    popops::mapInPlace(
        graph(),
        pe::Mul(pe::_1, pe::_2),
        {getInTensor(SGDVarUpdateOp::getVarToUpdateInIndex()),
         getInTensor(SGDVarUpdateOp::getWeightDecayScaleFactorInIndex())},
        prog,
        debugPrefix("nonConstWeightDecay"));
  }

  // const weight decay scale factor
  else {
    float scaleFactor = vu_op.initWeightDecayScaleFactor.val();
    if (scaleFactor != 1.0f) {
      popops::mapInPlace(graph(),
                         pe::Mul(pe::_1, pe::Const(scaleFactor)),
                         {getInTensor(SGDVarUpdateOp::getVarToUpdateInIndex())},
                         prog,
                         debugPrefix("constWeightDecay"));
    }
  }

  // (2) subtract scaled gradients
  poplar::Tensor weightDeltas =
      getInTensor(SGDVarUpdateOp::getUpdaterInIndex());

  if (dv_p->getReplicationFactor() > 1) {
    weightDeltas =
        popops::replicatedAllReduce(graph(),
                                    weightDeltas,
                                    popops::Operation::ADD,
                                    prog,
                                    debugPrefix("allReduce_Add"),
                                    {{"useReplicatedImplementation", "true"}});
  }

  // non-const scaled learning rate case
  if (!vu_op.initScaledLearningRate.isConst()) {
    popops::scaledSubtractFrom(
        graph(),
        getInTensor(SGDVarUpdateOp::getVarToUpdateInIndex()), // weights
        weightDeltas,                                         // weightDeltas
        getInTensor(SGDVarUpdateOp::getScaledLearningRateInIndex()),
        prog,
        debugPrefix("nonConstScaledSubtract"));
  }

  // const scaled learning rate case
  else {
    popops::scaledSubtractFrom(
        graph(),
        getInTensor(vu_op.getVarToUpdateInIndex()), // weights
        weightDeltas,                               // weightDeltas
        vu_op.initScaledLearningRate.val(),
        prog,
        debugPrefix("scaledSubtract"));
  }

  // output is a reference to the updated input
  setOutTensor(SGDVarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(SGDVarUpdateOp::getVarToUpdateInIndex()));
}

CopyVarUpdateOpx::CopyVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<CopyVarUpdateOp>(op, Onnx::CustomOperators::CopyVarUpdate);
}

void CopyVarUpdateOpx::grow(poplar::program::Sequence &prog) const {
  auto vu_op = getOp<CopyVarUpdateOp>();
  poplar::program::Copy copy(getInTensor(VarUpdateOp::getUpdaterInIndex()),
                             getInTensor(VarUpdateOp::getVarToUpdateInIndex()));
  prog.add(copy);

  // output is a reference to destination of the copy
  setOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(VarUpdateOp::getVarToUpdateInIndex()));
}

namespace {
OpxCreator<SGDVarUpdateOpx>
    sgdVarUpdateOpxCreator(Onnx::CustomOperators::SgdVarUpdate);
OpxCreator<CopyVarUpdateOpx>
    copyVarUpdateOpxCreator(Onnx::CustomOperators::CopyVarUpdate);
} // namespace

} // namespace popx
} // namespace popart
