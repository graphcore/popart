// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/adadeltaupdater.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/adadeltaupdaterx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

AdaDeltaUpdaterOpx::AdaDeltaUpdaterOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<AdaDeltaUpdaterOp>(op, Onnx::CustomOperators::AdaDeltaUpdater);
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void AdaDeltaUpdaterOpx::grow(poplar::program::Sequence &prog) const {

  // see adaptive.hpp for the equations implemented here

  auto &rmspropUpdaterOp = getOp<AdaDeltaUpdaterOp>();

  poplar::Tensor grad  = getInTensor(AdaDeltaUpdaterOp::getGradInIndex());
  poplar::Tensor accl1 = getInTensor(AdaDeltaUpdaterOp::getAccl1InIndex());
  poplar::Tensor accl2 = getInTensor(AdaDeltaUpdaterOp::getAccl2InIndex());

  std::vector<poplar::Tensor> tensors = {grad, accl1, accl2};

  pe::Any epsexpr(pe::Const(0.0f));
  if (rmspropUpdaterOp.initEps.isConst()) {
    epsexpr = pe::Const(rmspropUpdaterOp.initEps.val());
  } else {
    tensors.push_back(getInTensor(AdaDeltaUpdaterOp::getEpsInIndex()));
    epsexpr = pe::PlaceHolder(tensors.size());
  }

  // sqrt(Accl2 + eps) / sqrt(Accl1 + eps) * grad
  poplar::Tensor updater = popops::map(
      graph(),
      pe::Cast(pe::Mul(pe::Divide(
                           pe::Sqrt(pe::Add(
                               pe::Cast(pe::_3, accl1.elementType()), epsexpr)),
                           pe::Sqrt(pe::Add(pe::_2, epsexpr))),
                       pe::Cast(pe::_1, accl1.elementType())),
               grad.elementType()),
      tensors,
      prog,
      debugPrefix(""));

  if (hasInViewChangers(AdaDeltaUpdaterOp::getGradInIndex())) {
    setOutViewChangers(AdaDeltaUpdaterOp::getUpdaterOutIndex(),
                       getInViewChangers(AdaDeltaUpdaterOp::getGradInIndex()));
  }

  setOutTensor(AdaDeltaUpdaterOp::getUpdaterOutIndex(), updater);
}

poplar::Tensor AdaDeltaUpdaterOpx::createInput(int inIndex,
                                               const std::string &name) const {

  if (inIndex != AdaDeltaUpdaterOp::getAccl2InIndex()) {
    throw error("AccumulateOpx::createInput, cannot create input at {}, it can "
                "only create the var to update input Tensor",
                inIndex);
  }
  poplar::Tensor inTensor;
  auto accumulatorInfo = inInfo(inIndex);
  return graph().clone(popType(accumulatorInfo),
                       getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex()),
                       name);
}

InputCreatorType AdaDeltaUpdaterOpx::getInputCreatorType(int inIndex) const {
  return inIndex == AdaDeltaUpdaterOp::getAccl2InIndex()
             ? InputCreatorType::CanCreate
             : Opx::getInputCreatorType(inIndex);
}

std::vector<TensorId>
AdaDeltaUpdaterOpx::mustExistBeforeCreate(InIndex index) const {
  if (index != AdaDeltaUpdaterOp::getAccl2InIndex()) {
    throw internal_error(
        "AccumulateOpx::mustExistBeforeCreate : Invalid index");
  }
  return {inId(AdaDeltaUpdaterOp::getAccl1InIndex())};
}

bool AdaDeltaUpdaterOpx::hasCreatorViewChangers(InIndex index) const {
  return (index == AdaDeltaUpdaterOp::getAccl2InIndex()) &&
         hasInViewChangers(AdaDeltaUpdaterOp::getAccl1InIndex());
}

ViewChangers AdaDeltaUpdaterOpx::getCreatorViewChangers(InIndex index) const {
  if (index == AdaDeltaUpdaterOp::getAccl2InIndex()) {
    return getInViewChangers(AdaDeltaUpdaterOp::getAccl1InIndex());
  }
  throw error(
      "ReplicatedAllGatherOpx::getCreatorViewChangers: Invalid index = " +
      std::to_string(index));
}

namespace {
OpxCreator<AdaDeltaUpdaterOpx>
    AdaDeltaUpdaterOpxCreator(Onnx::CustomOperators::AdaDeltaUpdater);
}
} // namespace popx
} // namespace popart
