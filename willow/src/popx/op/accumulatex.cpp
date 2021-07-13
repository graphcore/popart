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
    : AccumulateBaseOpx(op, devicex) {
  verifyOp<AccumulateOp>(op, {Onnx::CustomOperators::Accumulate});
}

void AccumulateOpx::grow(poplar::program::Sequence &prog) const {

  auto &accumulateOp = getOp<AccumulateOp>();

  auto isConst = accumulateOp.getFactor().isConst();

  auto accum =
      getInTensor(VarUpdateOp::getVarToUpdateInIndex()).getPoplarTensor();

  auto grad = getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex())
                  .getPoplarTensor();

  // If the accl/accum tensor to update has a view changer,
  // but the updater does not, update the view instead
  // This may occur if the VarToUpdate tensor is CBR-rearranged
  // (see GCL CollectivesBalancedReorder.cpp)
  // (e.g. accumulator), but the Updater is not (e.g. gradient)
  if (hasInViewChangers(VarUpdateOp::getVarToUpdateInIndex()) &&
      !hasInViewChangers(VarUpdateWithUpdaterOp::getUpdaterInIndex())) {
    accum = getInView(VarUpdateOp::getVarToUpdateInIndex()).getPoplarTensor();
  }

  switch (accumulateOp.getAccumulationType()) {
  case AccumulationType::Add: {
    // accum += grad
    popops::scaledAddTo(graph().getPoplarGraph(),
                        accum,
                        grad,
                        1.0f,
                        prog,
                        debugContext("constAdd"));
    break;
  }
  case AccumulationType::Mean: {
    auto counter =
        getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();

    auto counter_1 = popops::add(graph().getPoplarGraph(),
                                 counter,
                                 1.0f,
                                 prog,
                                 debugContext("counter_1"));
    auto b         = popops::div(
        graph().getPoplarGraph(), 1.0f, counter_1, prog, debugContext("b"));
    auto a =
        popops::sub(graph().getPoplarGraph(), 1.0f, b, prog, debugContext("a"));

    popops::scaledAddTo(graph().getPoplarGraph(),
                        accum,
                        a,
                        grad,
                        b,
                        prog,
                        debugContext("Mean"));
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
        popops::scaledAddTo(graph().getPoplarGraph(),
                            accum,
                            grad,
                            1.0f,
                            prog,
                            debugContext("constAdd"));
      } else {
        // accum += factor * grad
        popops::scaledAddTo(graph().getPoplarGraph(),
                            accum,
                            grad,
                            val,
                            prog,
                            debugContext("constDampenedAdd"));
      }
    } else {
      auto factor =
          getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();
      popops::scaledAddTo(graph().getPoplarGraph(),
                          accum,
                          grad,
                          factor,
                          prog,
                          debugContext("dampenedAdd"));
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
            graph().getPoplarGraph(),
            pe::Add(pe::_1, pe::Square(pe::Cast(pe::_2, accum.elementType()))),
            {accum, grad},
            prog,
            debugContext("constAddSquare"));
      } else {
        auto val = accumulateOp.getFactor().val();
        // accum += factor * grad^2
        popops::mapInPlace(
            graph().getPoplarGraph(),
            pe::Add(pe::_1,
                    pe::Mul(pe::Mul(pe::Const(val),
                                    pe::Cast(pe::_2, accum.elementType())),
                            pe::Cast(pe::_2, accum.elementType()))),
            {accum, grad},
            prog,
            debugContext("constDampenedAddSquare"));
      }
    } else {
      auto factor =
          getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();
      popops::mapInPlace(
          graph().getPoplarGraph(),
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
      popops::mapInPlace(graph().getPoplarGraph(),
                         pe::Add(pe::Mul(pe::Const(val), pe::_1),
                                 pe::Cast(pe::_2, accum.elementType())),
                         {accum, grad},
                         prog,
                         debugContext("constDecayAdd"));
    } else {
      auto factor =
          getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();
      popops::mapInPlace(
          graph().getPoplarGraph(),
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
          graph().getPoplarGraph(),
          pe::Add(pe::Mul(pe::Const(val), pe::_1),
                  pe::Square(pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad},
          prog,
          debugContext("constDecayAddSquare"));
    } else {
      auto factor =
          getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();
      popops::mapInPlace(
          graph().getPoplarGraph(),
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
          graph().getPoplarGraph(),
          pe::Add(pe::Mul(pe::Const(val), pe::_1),
                  pe::Mul(pe::Const(1.0f - val),
                          pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad},
          prog,
          debugContext("constMovingAverage"));
    } else {
      auto factor =
          getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();
      popops::mapInPlace(
          graph().getPoplarGraph(),
          pe::Add(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                  pe::Mul(pe::Cast(pe::Sub(pe::Const(1.0f), pe::_3),
                                   accum.elementType()),
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
          graph().getPoplarGraph(),
          pe::Add(pe::Mul(pe::Const(val), pe::_1),
                  pe::Mul(pe::Mul(pe::Const(1.0f - val),
                                  pe::Cast(pe::_2, accum.elementType())),
                          pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad},
          prog,
          debugContext("constMovingAverageSquare"));
    } else {
      auto factor =
          getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();
      popops::mapInPlace(
          graph().getPoplarGraph(),
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
          graph().getPoplarGraph(),
          pe::Cast(pe::Max(pe::Mul(pe::Const(val), pe::_1),
                           pe::Cast(pe::Abs(pe::_2), accum.elementType())),
                   accum.elementType()),
          {accum, grad},
          prog,
          debugContext("constInfinity"));
    } else {
      auto factor =
          getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();
      popops::mapInPlace(
          graph().getPoplarGraph(),
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

snap::Tensor
AccumulateBaseOpx::createInputTensor(InIndex inIndex,
                                     const poplar::DebugNameAndId &dnai) const {
  if (inIndex != VarUpdateOp::getVarToUpdateInIndex()) {
    throw error(
        "AccumulateBaseOpx::createInput, cannot create input at {}, it can "
        "only create the var to update input Tensor",
        inIndex);
  }
  auto accumulatorInfo = inInfo(inIndex);
  return snap::Tensor{
      graph().getPoplarGraph().clone(
          popType(accumulatorInfo),
          getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex())
              .getPoplarTensor(),
          dnai),
      graph()};
}

InputCreatorType AccumulateBaseOpx::getInputCreatorType(int inIndex) const {
  return inIndex == VarUpdateOp::getVarToUpdateInIndex()
             ? InputCreatorType::CanCreate
             : PopOpx::getInputCreatorType(inIndex);
}

std::set<TensorId>
AccumulateBaseOpx::mustExistBeforeCreate(InIndex index) const {
  if (index != VarUpdateOp::getVarToUpdateInIndex()) {
    throw internal_error(
        "AccumulateBaseOpx::mustExistBeforeCreate : Invalid index");
  }
  return {inId(VarUpdateWithUpdaterOp::getUpdaterInIndex())};
}

bool AccumulateBaseOpx::hasCreatorViewChangers(InIndex index) const {
  return (index == VarUpdateOp::getVarToUpdateInIndex()) &&
         hasInViewChangers(VarUpdateWithUpdaterOp::getUpdaterInIndex());
}

ViewChangers AccumulateBaseOpx::getCreatorViewChangers(InIndex index) const {
  if (index == VarUpdateOp::getVarToUpdateInIndex()) {
    return getInViewChangers(VarUpdateWithUpdaterOp::getUpdaterInIndex());
  }
  throw error(
      "ReplicatedAllGatherOpx::getCreatorViewChangers: Invalid index = " +
      std::to_string(index));
}

RescaleAccumulateOpx::RescaleAccumulateOpx(Op *op, Devicex *devicex)
    : AccumulateBaseOpx(op, devicex) {
  verifyOp<RescaleAccumulateOp>(op, {Onnx::CustomOperators::RescaleAccumulate});
}

void RescaleAccumulateOpx::grow(poplar::program::Sequence &prog) const {

  auto &accumulateOp = getOp<RescaleAccumulateOp>();

  auto isConst = accumulateOp.getFactor().isConst();

  auto accum =
      getInTensor(VarUpdateOp::getVarToUpdateInIndex()).getPoplarTensor();

  auto grad = getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex())
                  .getPoplarTensor();

  auto rescaleRatio = getInTensor(RescaleAccumulateOp::getRescaleRatioInIndex())
                          .getPoplarTensor();

  // If the accl/accum tensor to update has a view changer,
  // but the updater does not, update the view instead
  // This may occur if the VarToUpdate tensor is CBR-rearranged
  // (see GCL CollectivesBalancedReorder.cpp)
  // (e.g. accumulator), but the Updater is not (e.g. gradient)
  if (hasInViewChangers(VarUpdateOp::getVarToUpdateInIndex()) &&
      !hasInViewChangers(VarUpdateWithUpdaterOp::getUpdaterInIndex())) {
    accum = getInView(VarUpdateOp::getVarToUpdateInIndex()).getPoplarTensor();
  }

  switch (accumulateOp.getAccumulationType()) {
  case AccumulationType::MovingAverage: {
    poplar::Tensor a, b;
    if (isConst) {
      auto val = accumulateOp.getFactor().val();
      a        = popops::mul(
          graph().getPoplarGraph(), rescaleRatio, val, prog, debugContext("a"));
      b = getConst(poplar::FLOAT, {}, 1.0f - val, "b").getPoplarTensor();
    } else {
      auto factor = getInTensor(RescaleAccumulateOp::getFactorInIndex())
                        .getPoplarTensor();
      a = popops::mul(graph().getPoplarGraph(),
                      rescaleRatio,
                      factor,
                      prog,
                      debugContext("a"));
      b = popops::sub(
          graph().getPoplarGraph(), 1.0f, factor, prog, debugContext("b"));
    }
    popops::scaledAddTo(graph().getPoplarGraph(),
                        accum,
                        a,
                        grad,
                        b,
                        prog,
                        debugContext("movingAverage"));
    break;
  }
  case AccumulationType::MovingAverageSquare: {
    if (isConst) {
      auto val = accumulateOp.getFactor().val();
      popops::mapInPlace(
          graph().getPoplarGraph(),
          pe::Add(pe::Mul(pe::_1, pe::Mul(pe::Const(val), pe::_3)),
                  pe::Mul(pe::Mul(pe::Const(1.0f - val),
                                  pe::Cast(pe::_2, accum.elementType())),
                          pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad, rescaleRatio},
          prog,
          debugContext("constMovingAverageSquare"));
    } else {
      auto factor = getInTensor(RescaleAccumulateOp::getFactorInIndex())
                        .getPoplarTensor();
      popops::mapInPlace(
          graph().getPoplarGraph(),
          pe::Add(
              pe::Mul(pe::Cast(pe::Mul(pe::_3, pe::_4), accum.elementType()),
                      pe::_1),
              pe::Mul(pe::Mul(pe::Sub(pe::Const(1.0f), pe::_4),
                              pe::Cast(pe::_2, accum.elementType())),
                      pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad, rescaleRatio, factor},
          prog,
          debugContext("movingAverageSquare"));
    }
    break;
  }
  case AccumulationType::Infinity: {
    if (isConst) {
      auto val = accumulateOp.getFactor().val();
      popops::mapInPlace(
          graph().getPoplarGraph(),
          pe::Max(pe::Mul(pe::Mul(pe::Const(val), pe::_3), pe::_1),
                  pe::Cast(pe::Abs(pe::_2), accum.elementType())),
          {accum, grad, rescaleRatio},
          prog,
          debugContext("constInfinity"));
    } else {
      auto factor = getInTensor(RescaleAccumulateOp::getFactorInIndex())
                        .getPoplarTensor();
      popops::mapInPlace(
          graph().getPoplarGraph(),
          pe::Max(
              pe::Mul(pe::Cast(pe::Mul(pe::_3, pe::_4), accum.elementType()),
                      pe::_1),
              pe::Cast(pe::Abs(pe::_2), accum.elementType())),
          {accum, grad, rescaleRatio, factor},
          prog,
          debugContext("infinity"));
    }
    break;
  }
  default:
    throw internal_error(
        "Unsupported AccumulationType in RescaleAccumulateOpx {}.",
        static_cast<int>(accumulateOp.getAccumulationType()));
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

namespace {
OpxCreator<AccumulateOpx>
    AccumulateOpxCreator({Onnx::CustomOperators::Accumulate});
OpxCreator<RescaleAccumulateOpx>
    RescaleAccumulateOpxCreator({Onnx::CustomOperators::RescaleAccumulate});
} // namespace

} // namespace popx
} // namespace popart
