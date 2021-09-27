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
#include <popart/popx/op/gatherx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/DynamicSlice.hpp>
#include <popops/Gather.hpp>

#include <set>

namespace pe = popops::expr;

namespace popart {
namespace popx {

/********** AccumulateBaseOpx **********/

AccumulateBaseOpx::AccumulateBaseOpx(Op *op, Devicex *devicex)
    : VarUpdateOpx(op, devicex) {}

InputCreatorType AccumulateBaseOpx::getInputCreatorType(int inIndex) const {
  return inIndex == VarUpdateOp::getVarToUpdateInIndex()
             ? InputCreatorType::CanCreate
             : PopOpx::getInputCreatorType(inIndex);
}

snap::Tensor
AccumulateBaseOpx::createInputTensor(int inIndex,
                                     const poplar::DebugNameAndId &dnai) const {

  if (inIndex != VarUpdateOp::getVarToUpdateInIndex()) {
    throw error(
        "AccumulateBaseOpx::createInput, cannot create input at {}, it can "
        "only create the var to update input Tensor",
        inIndex);
  }
  poplar::Tensor inTensor;
  auto accumulatorInfo = inInfo(inIndex);
  return snap::Tensor{
      graph().getPoplarGraph().clone(
          popType(accumulatorInfo),
          getInTensor(VarUpdateWithUpdaterOp::getUpdaterInIndex())
              .getPoplarTensor(),
          dnai),
      graph()};
}

std::set<TensorId>
AccumulateBaseOpx::mustExistBeforeCreate(InIndex index) const {
  if (index != VarUpdateOp::getVarToUpdateInIndex()) {
    throw internal_error(
        "AccumulateBaseOpx::mustExistBeforeCreate: Invalid index {}", index);
  }
  return {inId(VarUpdateWithUpdaterOp::getUpdaterInIndex())};
}

bool AccumulateBaseOpx::hasCreatorViewChangers(InIndex index) const {
  return (index == VarUpdateOp::getVarToUpdateInIndex()) &&
         hasInViewChangers(VarUpdateWithUpdaterOp::getUpdaterInIndex());
}

ViewChangers AccumulateBaseOpx::getCreatorViewChangers(InIndex index) const {
  if (index != AccumulateBaseOp::getVarToUpdateInIndex()) {
    throw error("AccumulateBaseOpx::getCreatorViewChangers: Invalid index = " +
                std::to_string(index));
  }

  return getInViewChangers(AccumulateBaseOp::getUpdaterInIndex());
}

/********** AccumulateOpx **********/

AccumulateOpx::AccumulateOpx(Op *op, Devicex *devicex)
    : AccumulateBaseOpx(op, devicex) {
  verifyOp<AccumulateOp>(op, {Onnx::CustomOperators::Accumulate});
}

void AccumulateOpx::grow(snap::program::Sequence &prog) const {

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
                        prog.getPoplarSequence(),
                        debugContext("constAdd"));
    break;
  }
  case AccumulationType::Mean: {
    auto counter =
        getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();

    auto counter_1 = popops::add(graph().getPoplarGraph(),
                                 counter,
                                 1.0f,
                                 prog.getPoplarSequence(),
                                 debugContext("counter_1"));
    auto b         = popops::div(graph().getPoplarGraph(),
                         1.0f,
                         counter_1,
                         prog.getPoplarSequence(),
                         debugContext("b"));
    auto a         = popops::sub(graph().getPoplarGraph(),
                         1.0f,
                         b,
                         prog.getPoplarSequence(),
                         debugContext("a"));

    popops::scaledAddTo(graph().getPoplarGraph(),
                        accum,
                        a,
                        grad,
                        b,
                        prog.getPoplarSequence(),
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
                            prog.getPoplarSequence(),
                            debugContext("constAdd"));
      } else {
        // accum += factor * grad
        popops::scaledAddTo(graph().getPoplarGraph(),
                            accum,
                            grad,
                            val,
                            prog.getPoplarSequence(),
                            debugContext("constDampenedAdd"));
      }
    } else {
      auto factor =
          getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();
      popops::scaledAddTo(graph().getPoplarGraph(),
                          accum,
                          grad,
                          factor,
                          prog.getPoplarSequence(),
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
            prog.getPoplarSequence(),
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
            prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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
                         prog.getPoplarSequence(),
                         debugContext("constDecayAdd"));
    } else {
      auto factor =
          getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();
      popops::mapInPlace(
          graph().getPoplarGraph(),
          pe::Add(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                  pe::Cast(pe::_2, accum.elementType())),
          {accum, grad, factor},
          prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
          debugContext("constDecayAddSquare"));
    } else {
      auto factor =
          getInTensor(AccumulateOp::getFactorInIndex()).getPoplarTensor();
      popops::mapInPlace(
          graph().getPoplarGraph(),
          pe::Add(pe::Mul(pe::Cast(pe::_3, accum.elementType()), pe::_1),
                  pe::Square(pe::Cast(pe::_2, accum.elementType()))),
          {accum, grad, factor},
          prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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

namespace {
OpxCreator<AccumulateOpx>
    AccumulateOpxCreator({Onnx::CustomOperators::Accumulate});
}

/********** SparseAccumulateOpx **********/

SparseAccumulateOpx::SparseAccumulateOpx(Op *op, Devicex *devicex)
    : AccumulateBaseOpx(op, devicex) {
  verifyOp<SparseAccumulateOp>(op, {Onnx::CustomOperators::SparseAccumulate});

  inputCreatorPriority = std::numeric_limits<double>::max();
}

snap::Tensor SparseAccumulateOpx::createInputTensor(
    InIndex inIndex,
    const poplar::DebugNameAndId &dnai) const {
  if (inIndex != SparseAccumulateOp::getVarToUpdateInIndex()) {
    throw error(
        "SparseAccumulateOpx::createInputTensor: Invalid input index {}",
        inIndex);
  }

  /*
    When choosing the tile layout of the input `accum` tensor, we generally have
    two options:
      1. Match the incoming gradient (the updater tensor), to avoid exchange
         inside this accumulate Op.
      2. Match the weight, to avoid exchange in the subsequent VarUpdate op.
    Typically, we always choose the former, as the accumulation happens inside
    the gradient accumulation loop (so it happens `af = accumulation factor`
    times, instead of once like the var update).

    However, for a SparseAccumulateOp, we cannot do 1 because the shape of the
    gradient is different from the shape of accum. Thus we use
    popops::createGatherInput to create a tensor with a layout that can be
    efficiently popops::multiUpdateAdd-ed into.

    --------

    Recall a "tied gather" operation, where in the forward pass two different
    views of the weight are used, once by the Gather this SparseAccumulate is
    for, and once by another Op. For example:

      x ---\
            \
      w ----> MatMul -----> y
        \
         ---> Transpose --> w^T --> Gather --> z

    In the backward pass: (Recall the GatherGrad -> Transpose -> Accumulate
    created in the backward pass are optimised into a single
    SparseAccumulate):

      y' ----> MatMul ----> dW_mm
             /                    \  accum
      x ----/                      \  |
                                  Accumulate
                                      |
                                    accum'
                                      |
      z' ---------------------> SparseAccumulate          w
                                      |                   |
                                    accum''  -------> VarUpdate
                                                          |
                                                          w'

    How do we lay out accum? We could:
      1. Match the updater from the first accumulate, dW_mm.
      2. Match the updater from the second accumulate, z'.
      3. Match the weight. Note this is the root weight not the transposed one.

    We choose to do 3. This is for historical reasons, when BERT was being run
    with continuous weight update pipelining.

    This behaviour is enabled by connecting the weight at
    SparseAccumulateOp::getOriginalVarToUpdateInIndex(). \sa SparseAccumulateOp.
    Otherwise, the usual popops::createGatherInput behaviour occurs.
   */

  if (hasInput(SparseAccumulateOp::getOriginalVarToUpdateInIndex())) {
    auto w = getInTensor(SparseAccumulateOp::getOriginalVarToUpdateInIndex())
                 .getPoplarTensor();
    return snap::Tensor{graph().getPoplarGraph().clone(w, dnai), graph()};
  }

  auto info        = inInfo(SparseAccumulateOp::getVarToUpdateInIndex());
  const auto shape = info.shape_szt();

  const auto &op = getOp<SparseAccumulateOp>();

  return snap::Tensor{popops::createGatherInput(graph().getPoplarGraph(),
                                                popx::popType(info),
                                                shape,
                                                op.getAxis(),
                                                popops::GatherParams{},
                                                dnai),
                      graph()};
}

void SparseAccumulateOpx::grow(snap::program::Sequence &prog) const {
  const auto op = getOp<SparseAccumulateOp>();

  const auto &initFactor = op.getFactor();
  const auto isConst     = initFactor.isConst();

  const auto axis = op.getAxis();

  auto accl = getInTensor(SparseAccumulateOp::getVarToUpdateInIndex())
                  .getPoplarTensor();
  auto grad =
      getInTensor(SparseAccumulateOp::getUpdaterInIndex()).getPoplarTensor();
  auto indices =
      getInTensor(SparseAccumulateOp::getIndicesInIndex()).getPoplarTensor();
  auto factor =
      isConst
          ? getConst(
                accl.elementType(), {}, initFactor.val(), "ConstSparseFactor")
                .getPoplarTensor()
          : getInTensor(SparseAccumulateOp::getFactorInIndex())
                .getPoplarTensor();

  // Rolls axis to front.
  const auto inputs =
      GatherGradOpx::handleNDMultiUpdate(accl, grad, indices, axis);
  auto &targetND  = std::get<0>(inputs);
  auto &updateND  = std::get<1>(inputs);
  auto &indicesND = std::get<2>(inputs);

  // Accumulate the updates into the target
  popops::multiUpdateAdd(graph().getPoplarGraph(),
                         targetND,
                         updateND,
                         indicesND,
                         factor,
                         {0},
                         {1},
                         prog.getPoplarSequence(),
                         popops::SlicePlan(),
                         poplar::OptionFlags(),
                         debugContext("nonConstSparseSGD1Accl"));

  // reference accl returned
  setOutTensor(SparseAccumulateOp::getUpdatedVarOutIndex(),
               snap::Tensor{accl, graph()});
}

std::set<TensorId>
SparseAccumulateOpx::mustExistBeforeCreate(InIndex inIndex) const {
  if (inIndex == SparseAccumulateOp::getVarToUpdateInIndex()) {
    if (hasInput(SparseAccumulateOp::getOriginalVarToUpdateInIndex())) {
      return {inId(SparseAccumulateOp::getOriginalVarToUpdateInIndex())};
    } else {
      return {};
    }
  }

  throw internal_error(
      "SparseAccumulateOpx::mustExistBeforeCreate: Invalid index {}", inIndex);
}

namespace {
OpxCreator<SparseAccumulateOpx>
    SparseAccumulateOpxCreator(Onnx::CustomOperators::SparseAccumulate);
} // namespace

/********** RescaleAccumulateOpx **********/

RescaleAccumulateOpx::RescaleAccumulateOpx(Op *op, Devicex *devicex)
    : AccumulateBaseOpx(op, devicex) {
  verifyOp<RescaleAccumulateOp>(op, {Onnx::CustomOperators::RescaleAccumulate});
}

void RescaleAccumulateOpx::grow(snap::program::Sequence &prog) const {

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
      a        = popops::mul(graph().getPoplarGraph(),
                      rescaleRatio,
                      val,
                      prog.getPoplarSequence(),
                      debugContext("a"));
      b        = getConst(poplar::FLOAT, {}, 1.0f - val, "b").getPoplarTensor();
    } else {
      auto factor = getInTensor(RescaleAccumulateOp::getFactorInIndex())
                        .getPoplarTensor();
      a = popops::mul(graph().getPoplarGraph(),
                      rescaleRatio,
                      factor,
                      prog.getPoplarSequence(),
                      debugContext("a"));
      b = popops::sub(graph().getPoplarGraph(),
                      1.0f,
                      factor,
                      prog.getPoplarSequence(),
                      debugContext("b"));
    }
    popops::scaledAddTo(graph().getPoplarGraph(),
                        accum,
                        a,
                        grad,
                        b,
                        prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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
          prog.getPoplarSequence(),
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
OpxCreator<RescaleAccumulateOpx>
    RescaleAccumulateOpxCreator({Onnx::CustomOperators::RescaleAccumulate});
} // namespace

} // namespace popx
} // namespace popart
