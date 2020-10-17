// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popops/Cast.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ScaledAdd.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/popx/devicex.hpp>
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
  auto grad  = getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex());

  switch (accumulateOp.getAccumulationType()) {
  case AccumulationType::Add: {
    // accum += grad
    popops::mapInPlace(graph(),
                       pe::Add(pe::_1, pe::Cast(pe::_2, accum.elementType())),
                       {accum, grad},
                       prog,
                       debugPrefix("constAdd"));
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
        popops::mapInPlace(
            graph(),
            pe::Add(pe::_1, pe::Cast(pe::_2, accum.elementType())),
            {accum, grad},
            prog,
            debugPrefix("constAdd"));
      } else {
        // accum += factor * grad
        popops::scaledAddTo(
            graph(), accum, grad, val, prog, debugPrefix("constDampenedAdd"));
      }
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::scaledAddTo(
          graph(), accum, grad, factor, prog, debugPrefix("dampenedAdd"));
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
            debugPrefix("constAddSquare"));
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
            debugPrefix("constDampenedAddSquare"));
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
          debugPrefix("dampenedAddSquare"));
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
                         debugPrefix("constDecayAdd"));
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::mapInPlace(
          graph(),
          pe::Add(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                  pe::Cast(pe::_2, accum.elementType())),
          {accum, grad, factor},
          prog,
          debugPrefix("decayAdd"));
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
          debugPrefix("constDecayAddSquare"));
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::mapInPlace(
          graph(),
          pe::Add(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                  pe::Square(pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad, factor},
          prog,
          debugPrefix("decayAddSquare"));
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
          debugPrefix("constMovingAverage"));
    } else {
      auto factor = getInTensor(AccumulateOp::getFactorInIndex());
      popops::mapInPlace(
          graph(),
          pe::Add(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                  pe::Mul(pe::Sub(pe::Const(1.0f), pe::_3),
                          pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad, factor},
          prog,
          debugPrefix("movingAverage"));
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
          debugPrefix("constMovingAverageSquare"));
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
          debugPrefix("movingAverageSquare"));
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
          debugPrefix("constInfinity"));
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
          debugPrefix("infinity"));
    }
    break;
  }
  }

  if (hasInViewChangers(VarUpdateWithUpdaterOp::getUpdaterInIndex())) {
    setOutViewChangers(
        VarUpdateOp::getUpdatedVarOutIndex(),
        getInViewChangers(VarUpdateWithUpdaterOp::getUpdaterInIndex()));
  }
  // reference accum returned
  setOutTensor(VarUpdateOp::getUpdatedVarOutIndex(), accum);
}

poplar::Tensor AccumulateOpx::createInput(int inIndex,
                                          const std::string &name) const {

  if (inIndex != VarUpdateOp::getVarToUpdateInIndex()) {
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

InputCreatorType AccumulateOpx::getInputCreatorType(int inIndex) const {
  return inIndex == VarUpdateOp::getVarToUpdateInIndex()
             ? InputCreatorType::CanCreate
             : Opx::getInputCreatorType(inIndex);
}

std::vector<TensorId>
AccumulateOpx::mustExistBeforeCreate(InIndex index) const {
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
