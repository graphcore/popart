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

#include <poputil/TileMapping.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/operators.hpp"

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
  verifyOp<ScatterReduceOp>(op,
                            {Onnx::CustomOperators::ScatterReduce,
                             Onnx::Operators::Scatter_9,
                             Onnx::Operators::Scatter_11,
                             Onnx::Operators::ScatterElements_11});

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
  const auto &data    = getInTensor(srop.srcDataInIndex());
  const auto &indices = getInTensor(srop.indicesInIndex());

  const auto outIdx = srop.outIndex();
  poplar::Tensor out =
      createDataTensor(graph(),
                       outInfo(outIdx),
                       plan,
                       axis,
                       group_size,
                       srop.indexBroadcasted(),
                       getDebugNameAndId("scatterreduceOutput"));
  const auto initialValuesIdx = srop.initialValuesInIndex();

  if (srop.hasInput(initialValuesIdx)) {
    const auto &t = getInTensor(initialValuesIdx);
    prog.add(poplar::program::Copy(
        t, out, false, debugContext("copyToScatterReduce")));
  } else {
    strategy->initReductionOutput(*this, out, prog);
  }
  strategy->forward(
      srop, *this, out, data, indices, axis, group_size, prog, plan);
  if (group_size > 1) {
    const poplar::Tensor remapped_result =
        graph().addVariable(out.elementType(), out.shape(), "RemappedOutput");
    for (int g = 0; g < group_size; g++)
      poputil::mapTensorLinearly(graph(), remapped_result.slice(g, g + 1, 0));
    prog.add(poplar::program::Copy(out, remapped_result));
    setOutTensor(outIdx, remapped_result);
  } else {
    setOutTensor(outIdx, out);
  }
}

poplar::Tensor
ScatterReduceOpx::createInput(InIndex index,
                              const poplar::DebugNameAndId &dnai) const {
  auto &srop            = getOp<ScatterReduceOp>();
  const auto srcDataIdx = srop.srcDataInIndex();
  const auto indicesIdx = srop.indicesInIndex();

  if (index != srcDataIdx && index != indicesIdx) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  logging::debug("ScatterReduceOpx::createInput index={}", index);

  const auto indicesInfo = inInfo(indicesIdx);

  if (index == indicesIdx) {
    return createIndicesTensor(graph(),
                               indicesInfo,
                               plan,
                               axis,
                               group_size,
                               srop.indexBroadcasted(),
                               dnai);
  }

  return createUpdateTensor(graph(),
                            inInfo(srcDataIdx),
                            indicesInfo,
                            plan,
                            axis,
                            group_size,
                            srop.indexBroadcasted(),
                            dnai);
}

InputCreatorType ScatterReduceOpx::getInputCreatorType(InIndex index) const {
  const auto &srop = getOp<ScatterReduceOp>();

  if (index == srop.srcDataInIndex() || index == srop.indicesInIndex()) {
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
  if (group_size > 1) {
    const poplar::Tensor remapped_result = graph().addVariable(
        gradOut[0].elementType(), gradOut[0].shape(), "RemappedGradOutput");
    for (int g = 0; g < group_size; g++)
      poputil::mapTensorLinearly(graph(), remapped_result.slice(g, g + 1, 0));
    prog.add(poplar::program::Copy(gradOut[0], remapped_result));
    setOutTensor(ScatterReduceGradOp::gradDataOutIndex(), remapped_result);
  } else {
    setOutTensor(ScatterReduceGradOp::gradDataOutIndex(), gradOut[0]);
  }

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
