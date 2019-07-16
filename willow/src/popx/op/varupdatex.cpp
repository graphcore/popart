#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/op/varupdate.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/varupdatex.hpp>
#include <poponnx/popx/opxmanager.hpp>

#include <popops/Collectives.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>

namespace pe = popops::expr;

namespace poponnx {
namespace popx {

VarUpdateOpx::VarUpdateOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {}

poplar::Tensor VarUpdateOpx::createInput(int index,
                                         const std::string &name) const {
  if (index != VarUpdateOp::getUpdaterInIndex()) {
    throw error("VarUpdateOpx::createInput Cannot create input {}", index);
  }

  poplar::Tensor var = getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  return graph().clone(var, name);
}

InputCreatorType VarUpdateOpx::getInputCreatorType(int index1) const {
  return index1 == VarUpdateOp::getUpdaterInIndex() &&
                 op_p->getIr().getSessionOptions().enableGradientAccumulation
             ? InputCreatorType::CANCREATE
             : Opx::getInputCreatorType(index1);
}

std::vector<TensorId> VarUpdateOpx::mustExistBeforeCreate(int index1) const {
  if (index1 != VarUpdateOp::getUpdaterInIndex()) {
    throw error("VarUpdateOpx::mustExistBeforeCreate : Invalid index = " +
                std::to_string(index1));
  }

  return {inId(VarUpdateOp::getVarToUpdateInIndex())};
}

SGDVarUpdateOpx::SGDVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<SGDVarUpdateOp>(op, Onnx::CustomOperators::SgdVarUpdate);
}

void SGDVarUpdateOpx::grow(poplar::program::Sequence &prog) const {

  // Weight update (matching pytorch implementation):
  //   w <- w * (1 - lr * wd) - lr * delta

  // The (1 -lr * wd) calculation is done in SGD::setTensorData, weightDecay is
  // weightDecayScaleFactor

  // First update weights with weight decay
  popops::mapInPlace(graph(),
                     pe::Mul(pe::_1, pe::_2),
                     {getInTensor(SGDVarUpdateOp::getVarToUpdateInIndex()),
                      getInTensor(SGDVarUpdateOp::getWeightDecayInIndex())},
                     prog,
                     debugPrefix("weightDecay"));

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

  // Then subtract scaled gradients
  popops::scaledSubtractFrom(
      graph(),
      getInTensor(SGDVarUpdateOp::getVarToUpdateInIndex()), // weights
      weightDeltas,                                         // weightDeltas
      getInTensor(SGDVarUpdateOp::getLearnRateInIndex()),
      prog,
      debugPrefix("scaledSubtract"));

  // output is a reference to the updated input
  setOutTensor(SGDVarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(SGDVarUpdateOp::getVarToUpdateInIndex()));
}

ConstSGDVarUpdateOpx::ConstSGDVarUpdateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<ConstSGDVarUpdateOp>(op, Onnx::CustomOperators::ConstSgdVarUpdate);
}

void ConstSGDVarUpdateOpx::grow(poplar::program::Sequence &prog) const {
  auto vu_op = getOp<ConstSGDVarUpdateOp>();

  // Weight update (matching pytorch implementation):
  //   w <- w * (1 - lr * wd) - lr * delta

  // First update weights with weight decay (only if user has
  // specified non-zero weight decay)
  if (vu_op.getWeightDecay() != 0.0f) {
    float weightDecayScaleFactor =
        1 - (vu_op.getWeightDecay() * vu_op.getLearnRate());

    popops::mapInPlace(
        graph(),
        pe::Mul(pe::_1, pe::Const(weightDecayScaleFactor)),
        {getInTensor(ConstSGDVarUpdateOp::getVarToUpdateInIndex())},
        prog,
        debugPrefix("weightDecay"));
  }

  poplar::Tensor weightDeltas = getInTensor(vu_op.getUpdaterInIndex());

  if (dv_p->getReplicationFactor() > 1) {

    weightDeltas =
        popops::replicatedAllReduce(graph(),
                                    weightDeltas,
                                    popops::Operation::ADD,
                                    prog,
                                    debugPrefix("allReduce_Add"),
                                    {{"useReplicatedImplementation", "true"}});
  }

  // Then subtract scaled gradients
  popops::scaledSubtractFrom(
      graph(),
      getInTensor(vu_op.getVarToUpdateInIndex()), // weights
      weightDeltas,                               // weightDeltas
      vu_op.getLearnRate(),
      prog,
      debugPrefix("scaledSubtract"));

  // output is a reference to the updated input
  setOutTensor(ConstSGDVarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(ConstSGDVarUpdateOp::getVarToUpdateInIndex()));
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
OpxCreator<ConstSGDVarUpdateOpx>
    constSgdVarUpdateOpxCreator(Onnx::CustomOperators::ConstSgdVarUpdate);
OpxCreator<CopyVarUpdateOpx>
    copyVarUpdateOpxCreator(Onnx::CustomOperators::CopyVarUpdate);
} // namespace

} // namespace popx
} // namespace poponnx
