// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/accumulatex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

AccumulateOpx::AccumulateOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {
  verifyOp<AccumulateOp>(op, {Onnx::CustomOperators::Accumulate});
}

void AccumulateOpx::grow(poplar::program::Sequence &prog) const {

  auto &accumulateOp = getOp<AccumulateOp>();

  auto isConst = accumulateOp.getFactor().isConst();

  auto accum = getInTensor(VarUpdateOp::getVarToUpdateInIndex());

  auto grad = getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  // If the accl/accum tensor to update has a view changer,
  // but the updater does not, update the view instead
  // This may occur if the VarToUpdate tensor is CBR-rearranged
  // (see GCL CollectivesBalancedReorder.cpp)
  // (e.g. accumulator), but the Updater is not (e.g. gradient)
  if (hasInViewChangers(VarUpdateOp::getVarToUpdateInIndex()) &&
      !hasInViewChangers(VarUpdateWithUpdaterOp::getUpdaterInIndex())) {
    accum = getInView(VarUpdateOp::getVarToUpdateInIndex());
  }

  switch (accumulateOp.getAccumulationType()) {
  case AccumulationType::Add: {
    // accum += grad
    popops::scaledAddTo(
        graph(), accum, grad, 1.0f, prog, debugContext("constAdd"));
    break;
  }
  case AccumulationType::Mean: {
    poplar::Tensor counter = getInTensor(AccumulateOp::getFactorInIndex());

    auto counter_1 = popops::add(graph(), counter, 1.0f, prog);
    auto a         = popops::div(graph(), counter, counter_1, prog);
    auto b         = popops::div(graph(), 1.0f, counter_1, prog);

    popops::scaledAddTo(
        graph(), accum, a, grad, b, prog, debugContext("constAdd"));
    break;
  }
  case AccumulationType::DampenedAdd: {
    if (isConst) {
      auto val = accumulateOp.getFactor().val();
      if (val == 0.0f) {
        throw internal_error(
            "factor of 0 is not allowed, should have been caught in "
            "the Ir, factor of 0 could be caused by dampening of 1, which "
            "means the gradient is multiplied by 0 (no learning)");
      }
      if (val - 1.0f == 0.0f) {
        // accum += grad
        popops::scaledAddTo(
            graph(), accum, grad, 1.0f, prog, debugContext("constAdd"));
      } else {
        // accum += factor * grad
        popops::scaledAddTo(
            graph(), accum, grad, val, prog, debugContext("constDampenedAdd"));
      }
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::scaledAddTo(
          graph(), accum, grad, factor, prog, debugContext("dampenedAdd"));
    }
    break;
  }
  case AccumulationType::DampenedAddSquare: {
    if (isConst) {
      auto val = accumulateOp.getFactor().val();
      if (val == 0.0f) {
        throw internal_error(
            "factor of 0 is not allowed, should have been caught in "
            "the Ir, factor of 0 could be caused by dampening of 1, which "
            "means the gradient is multiplied by 0 (no learning)");
      }
      if (val - 1.0f == 0.0f) {
        // accum += grad^2
        popops::mapInPlace(
            graph(),
            pe::Add(pe::_1, pe::Square(pe::Cast(pe::_2, accum.elementType()))),
            {accum, grad},
            prog,
            debugContext("constAddSquare"));
      } else {
        auto val = accumulateOp.getFactor().val();
        // accum += factor * grad^2
        popops::mapInPlace(
            graph(),
            pe::Add(pe::_1,
                    pe::Mul(pe::Mul(pe::Const(val),
                                    pe::Cast(pe::_2, accum.elementType())),
                            pe::Cast(pe::_2, accum.elementType()))),
            {accum, grad},
            prog,
            debugContext("constDampenedAddSquare"));
      }
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::mapInPlace(
          graph(),
          pe::Add(
              pe::_1,
              pe::Mul(pe::Mul(pe::_3, pe::Cast(pe::_2, accum.elementType())),
                      pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad, factor},
          prog,
          debugContext("dampenedAddSquare"));
    }
    break;
  }
  case AccumulationType::DecayAdd: {
    if (isConst) {
      auto val = accumulateOp.getFactor().val();
      popops::mapInPlace(graph(),
                         pe::Add(pe::Mul(pe::Const(val), pe::_1),
                                 pe::Cast(pe::_2, accum.elementType())),
                         {accum, grad},
                         prog,
                         debugContext("constDecayAdd"));
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::mapInPlace(
          graph(),
          pe::Add(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                  pe::Cast(pe::_2, accum.elementType())),
          {accum, grad, factor},
          prog,
          debugContext("decayAdd"));
    }
    break;
  }
  case AccumulationType::DecayAddSquare: {
    if (isConst) {
      auto val = accumulateOp.getFactor().val();
      popops::mapInPlace(
          graph(),
          pe::Add(pe::Mul(pe::Const(val), pe::_1),
                  pe::Square(pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad},
          prog,
          debugContext("constDecayAddSquare"));
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::mapInPlace(
          graph(),
          pe::Add(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                  pe::Square(pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad, factor},
          prog,
          debugContext("decayAddSquare"));
    }
    break;
  }
  case AccumulationType::MovingAverage: {
    if (isConst) {
      auto val = accumulateOp.getFactor().val();
      popops::mapInPlace(
          graph(),
          pe::Add(pe::Mul(pe::Const(val), pe::_1),
                  pe::Mul(pe::Const(1.0f - val),
                          pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad},
          prog,
          debugContext("constMovingAverage"));
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::mapInPlace(
          graph(),
          pe::Add(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                  pe::Mul(pe::Sub(pe::Const(1.0f), pe::_3),
                          pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad, factor},
          prog,
          debugContext("movingAverage"));
    }
    break;
  }
  case AccumulationType::MovingAverageSquare: {
    if (isConst) {
      auto val = accumulateOp.getFactor().val();
      popops::mapInPlace(
          graph(),
          pe::Add(pe::Mul(pe::Const(val), pe::_1),
                  pe::Mul(pe::Mul(pe::Const(1.0f - val),
                                  pe::Cast(pe::_2, accum.elementType())),
                          pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad},
          prog,
          debugContext("constMovingAverageSquare"));
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::mapInPlace(
          graph(),
          pe::Add(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                  pe::Mul(pe::Mul(pe::Sub(pe::Const(1.0f), pe::_3),
                                  pe::Cast(pe::_2, accum.elementType())),
                          pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad, factor},
          prog,
          debugContext("movingAverageSquare"));
    }
    break;
  }
  case AccumulationType::Infinity: {
    if (isConst) {
      auto val = accumulateOp.getFactor().val();
      popops::mapInPlace(
          graph(),
          pe::Cast(pe::Max(pe::Mul(pe::Const(val), pe::_1),
                           pe::Cast(pe::Abs(pe::_2), accum.elementType())),
                   accum.elementType()),
          {accum, grad},
          prog,
          debugContext("constInfinity"));
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::mapInPlace(
          graph(),
          pe::Cast(
              pe::Max(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                      pe::Cast(pe::Abs(pe::_2), accum.elementType())),
              accum.elementType()),
          {accum, grad, factor},
          prog,
          debugContext("infinity"));
    }
    break;
  }
  }

  if (hasInViewChangers(VarUpdateWithUpdaterOp::getVarToUpdateInIndex())) {
    setOutViewChangers(
        VarUpdateOp::getUpdatedVarOutIndex(),
        getInViewChangers(VarUpdateWithUpdaterOp::getVarToUpdateInIndex()));
  }
  // reference accum returned (as tensor, including view changers)
  setOutTensor(VarUpdateOp::getUpdatedVarOutIndex(),
               getInTensor(VarUpdateWithUpdaterOp::getVarToUpdateInIndex()));
}

poplar::Tensor
AccumulateOpx::createInput(int inIndex,
                           const poplar::DebugNameAndId &dnai) const {

  if (inIndex != VarUpdateOp::getVarToUpdateInIndex()) {
    throw error("AccumulateOpx::createInput, cannot create input at {}, it can "
                "only create the var to update input Tensor",
                inIndex);
  }
  poplar::Tensor inTensor;
  auto accumulatorInfo = inInfo(inIndex);
  return graph().clone(popType(accumulatorInfo),
                       getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex()),
                       dnai);
}

InputCreatorType AccumulateOpx::getInputCreatorType(int inIndex) const {
  return inIndex == VarUpdateOp::getVarToUpdateInIndex()
             ? InputCreatorType::CanCreate
             : Opx::getInputCreatorType(inIndex);
}

std::set<TensorId> AccumulateOpx::mustExistBeforeCreate(InIndex index) const {
  if (index != VarUpdateOp::getVarToUpdateInIndex()) {
    throw internal_error(
        "AccumulateOpx::mustExistBeforeCreate : Invalid index");
  }
  return {inId(VarUpdateWithUpdaterOp::getUpdaterInIndex())};
}

bool AccumulateOpx::hasCreatorViewChangers(InIndex index) const {
  return (index == VarUpdateOp::getVarToUpdateInIndex()) &&
         hasInViewChangers(VarUpdateWithUpdaterOp::getUpdaterInIndex());
}

ViewChangers AccumulateOpx::getCreatorViewChangers(InIndex index) const {
  if (index == VarUpdateOp::getVarToUpdateInIndex()) {
    return getInViewChangers(VarUpdateWithUpdaterOp::getUpdaterInIndex());
  }
  throw error(
      "ReplicatedAllGatherOpx::getCreatorViewChangers: Invalid index = " +
      std::to_string(index));
}

namespace {
OpxCreator<AccumulateOpx>
    AccumulateOpxCreator({Onnx::CustomOperators::Accumulate});
}

} // namespace popx
} // namespace popart
