// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <ext/new_allocator.h>
#include <limits>
#include <memory>

#include <popart/op/scatterreduce.hpp>
#include <popart/popx/op/scatterreduceutilx.hpp>
#include <popart/popx/op/scatterreducex.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

namespace {

std::unique_ptr<scatterreduceutilx::IScatterReductionStrategy>
createStrategy(const ScatterReduction &reduction) {
  if (reduction == ScatterReduction::Sum) {
    return std::make_unique<scatterreduceutilx::SumReductionStrategy>();
  }
  if (reduction == ScatterReduction::Max) {
    return std::make_unique<scatterreduceutilx::MaxReductionStrategy>();
  }
  if (reduction == ScatterReduction::Min) {
    return std::make_unique<scatterreduceutilx::MinReductionStrategy>();
  }
  if (reduction == ScatterReduction::Mul) {
    return std::make_unique<scatterreduceutilx::MulReductionStrategy>();
  }
  if (reduction == ScatterReduction::None) {
    return std::make_unique<scatterreduceutilx::NoneReductionStrategy>();
  }
  throw popart::internal_error("Unsupported reduction strategy!");
}

} // namespace

ScatterReduceOpx::ScatterReduceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex), strategy(), plan(), axis() {
  verifyOp<ScatterReduceOp>(op, {Onnx::CustomOperators::ScatterReduce});

  const auto &srop   = getOp<ScatterReduceOp>();
  strategy           = createStrategy(srop.getReduction());
  const auto options = strategy->createForwardPlanOptions(srop);

  axis       = static_cast<size_t>(srop.getAxis());
  group_size = static_cast<size_t>(srop.getGroupSize());
  nonstd::optional<size_t> plan_axis =
      srop.indexBroadcasted() ? nonstd::optional<size_t>() : axis;

  plan = createSlicePlan(graph(),
                         outInfo(srop.outIndex()),
                         inInfo(srop.indicesInIndex()),
                         options,
                         plan_axis,
                         group_size);

  // We always want the ScatterReduce to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

ScatterReduceOpx::~ScatterReduceOpx() = default;

void ScatterReduceOpx::grow(poplar::program::Sequence &prog) const {
  const auto &srop    = getOp<ScatterReduceOp>();
  const auto &data    = getInTensor(ScatterReduceOp::dataInIndex());
  const auto &indices = getInTensor(ScatterReduceOp::indicesInIndex());

  poplar::Tensor out =
      createDataTensor(graph(),
                       outInfo(ScatterReduceOp::outIndex()),
                       plan,
                       axis,
                       group_size,
                       srop.indexBroadcasted(),
                       getDebugNameAndId("scatterreduceOutput"));

  if (srop.hasInput(ScatterReduceOp::initialValuesInIndex())) {
    const auto &t = getInTensor(ScatterReduceOp::initialValuesInIndex());
    prog.add(poplar::program::Copy(
        t, out, false, debugContext("copyToScatterReduce")));
  } else {
    strategy->initReductionOutput(*this, out, prog);
  }
  strategy->forward(
      srop, *this, out, data, indices, axis, group_size, prog, plan);
  setOutTensor(ScatterReduceOp::outIndex(), out);
}

poplar::Tensor
ScatterReduceOpx::createInput(InIndex index,
                              const poplar::DebugNameAndId &dnai) const {
  if (index != ScatterReduceOp::dataInIndex() &&
      index != ScatterReduceOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  logging::debug("ScatterReduceOpx::createInput index={}", index);

  auto &srop             = getOp<ScatterReduceOp>();
  const auto indicesInfo = inInfo(ScatterReduceOp::indicesInIndex());

  if (index == ScatterReduceOp::indicesInIndex()) {
    return createIndicesTensor(graph(),
                               indicesInfo,
                               plan,
                               axis,
                               group_size,
                               srop.indexBroadcasted(),
                               dnai);
  }

  return createUpdateTensor(graph(),
                            inInfo(ScatterReduceOp::dataInIndex()),
                            indicesInfo,
                            plan,
                            axis,
                            group_size,
                            srop.indexBroadcasted(),
                            dnai);
}

InputCreatorType ScatterReduceOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterReduceOp::dataInIndex() ||
      index == ScatterReduceOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return Opx::getInputCreatorType(index);
}

ScatterReduceGradOpx::ScatterReduceGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex), strategy(), plan(), axis(), group_size() {
  verifyOp<ScatterReduceGradOp>(
      op, {Onnx::CustomGradOperators::ScatterReduceGradOp});

  const auto &srop   = getOp<ScatterReduceGradOp>();
  strategy           = createStrategy(srop.getReduction());
  const auto options = strategy->createBackwardPlanOptions(srop);

  axis       = static_cast<size_t>(srop.getAxis());
  group_size = static_cast<size_t>(srop.getGroupSize());
  nonstd::optional<size_t> plan_axis =
      srop.indexBroadcasted() ? nonstd::optional<size_t>() : axis;

  plan = createSlicePlan(graph(),
                         inInfo(srop.gradInIndex()),
                         inInfo(srop.indicesInIndex()),
                         options,
                         plan_axis,
                         group_size);

  // We always want the ScatterReduceGrad to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

ScatterReduceGradOpx::~ScatterReduceGradOpx() = default;

void ScatterReduceGradOpx::grow(poplar::program::Sequence &prog) const {
  const auto &srop = getOp<ScatterReduceGradOp>();
  auto gradIn      = getInTensor(ScatterReduceGradOp::gradInIndex());
  auto indices     = getInTensor(ScatterReduceGradOp::indicesInIndex());

  if (!srop.indexBroadcastEnabled() && !srop.indexBroadcasted()) {
    throw error("ScatterReduceGradOpx: The backward pass is implemented only "
                "for src.shape == index.shape.");
  }

  const auto gradOut = strategy->backward(
      srop, *this, gradIn, indices, axis, group_size, prog, plan);

  if (gradOut.size() != srop.outTensorCount()) {
    throw error("ScatterReduceGradOpx must calculate at least one gradient "
                " tensor and no more than two.");
  }

  setOutTensor(ScatterReduceGradOp::gradDataOutIndex(), gradOut[0]);

  if (srop.hasInitialValues()) {
    setOutTensor(ScatterReduceGradOp::gradInitialValuesOutIndex(), gradOut[1]);
  }
}

poplar::Tensor
ScatterReduceGradOpx::createInput(InIndex index,
                                  const poplar::DebugNameAndId &dnai) const {
  if (index != ScatterReduceGradOp::gradInIndex() &&
      index != ScatterReduceGradOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  const auto &srop = getOp<ScatterReduceGradOp>();
  if (index == ScatterReduceGradOp::gradInIndex()) {
    return createDataTensor(graph(),
                            inInfo(index),
                            plan,
                            axis,
                            group_size,
                            srop.indexBroadcasted(),
                            dnai);
  }

  return createIndicesTensor(graph(),
                             inInfo(index),
                             plan,
                             axis,
                             group_size,
                             srop.indexBroadcasted(),
                             dnai);
}

InputCreatorType
ScatterReduceGradOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterReduceGradOp::gradInIndex() ||
      index == ScatterReduceGradOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }
  return Opx::getInputCreatorType(index);
}

namespace {
OpxCreator<ScatterReduceOpx>
    scatterReduceOpxCreator(Onnx::CustomOperators::ScatterReduce);
OpxCreator<ScatterReduceGradOpx>
    scatterReduceGradOpxCreator(Onnx::CustomGradOperators::ScatterReduceGradOp);
} // namespace

} // namespace popx
} // namespace popart
