// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <cstdint>
#include <ext/new_allocator.h>
#include <limits>
#include <vector>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Fill.hpp>
#include <popart/error.hpp>
#include <popart/op/scatter.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/scatterx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensorinfo.hpp"

namespace poplar {
class Graph;
} // namespace poplar

namespace popart {
class Op;

namespace popx {
class Devicex;
} // namespace popx
} // namespace popart

namespace popart {
namespace popx {

ScatterOpx::ScatterOpx(Op *op, Devicex *devicex)
    : ScatterReduceOpx(op, devicex) {}

ScatterDataGradOpx::ScatterDataGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ScatterDataGradOp>(op, Onnx::GradOperators::ScatterDataGrad);
  auto grad_op = getOp<ScatterDataGradOp>();
  axis         = grad_op.getAxis();
  auto options = createSlicePlanOptions(SlicePlanUsedFor::Update,
                                        grad_op.getAvailableMemoryProportion());
  plan         = createSlicePlan(graph(),
                         outInfo(grad_op.gradOutIndex()),
                         inInfo(grad_op.indicesInIndex()),
                         options);

  // We always want this op to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterDataGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradIn      = getInTensor(ScatterDataGradOp::gradInIndex());
  auto indices     = getInTensor(ScatterDataGradOp::indicesInIndex());
  auto gradInInfo  = inInfo(ScatterDataGradOp::gradInIndex());
  auto indicesInfo = inInfo(ScatterDataGradOp::indicesInIndex());
  auto uaxis       = static_cast<unsigned>(axis);

  auto zerosUpdate = createUpdateTensor(graph(),
                                        gradInInfo,
                                        indicesInfo,
                                        plan,
                                        uaxis,
                                        1U,
                                        /*broadcasted=*/true,
                                        getDebugNameAndId("zerosUpdate"));
  popops::fill(
      graph(), zerosUpdate, prog, 0.0f, debugContext("zerosUpdateFill"));

  auto gradOut = scatterutilx::growScatter(*this,
                                           prog,
                                           graph(),
                                           gradIn,
                                           zerosUpdate,
                                           indices,
                                           gradInInfo,
                                           plan,
                                           axis);

  setOutTensor(ScatterDataGradOp::gradOutIndex(), gradOut);
}

poplar::Tensor
ScatterDataGradOpx::createInput(InIndex index,
                                const poplar::DebugNameAndId &dnai) const {
  if (index != ScatterDataGradOp::indicesInIndex()) {
    throw error("ScatterDataGradOpx::createInput : Invalid index = {}", index);
  }

  auto indicesInfo = inInfo(ScatterDataGradOp::indicesInIndex());
  return createIndicesTensor(graph(), indicesInfo, plan, axis, 1U, true, dnai);
}

InputCreatorType ScatterDataGradOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterDataGradOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return Opx::getInputCreatorType(index);
}

ScatterUpdateGradOpx::ScatterUpdateGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ScatterUpdateGradOp>(op, Onnx::GradOperators::ScatterUpdateGrad);

  const auto &grad_op = getOp<ScatterUpdateGradOp>();
  axis                = grad_op.getAxis();
  auto options        = createSlicePlanOptions(SlicePlanUsedFor::Slice,
                                        grad_op.getAvailableMemoryProportion());
  plan                = createSlicePlan(graph(),
                         inInfo(grad_op.gradInIndex()),
                         inInfo(grad_op.indicesInIndex()),
                         options);

  // We always want this op to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterUpdateGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradOut = scatterutilx::growScatterUpdateGrad(
      *this,
      prog,
      graph(),
      getInTensor(ScatterUpdateGradOp::gradInIndex()),
      getInTensor(ScatterUpdateGradOp::indicesInIndex()),
      outInfo(ScatterUpdateGradOp::gradOutIndex()).shape(),
      axis,
      plan,
      getDebugNameAndId("scatterUpdateGrad"));

  setOutTensor(ScatterUpdateGradOp::gradOutIndex(), gradOut);
}

poplar::Tensor
ScatterUpdateGradOpx::createInput(InIndex index,
                                  const poplar::DebugNameAndId &dnai) const {

  if (index != ScatterUpdateGradOp::gradInIndex() &&
      index != ScatterUpdateGradOp::indicesInIndex()) {
    throw error("ScatterUpdateOpx::createInput : Invalid index = {}", index);
  }

  if (index == ScatterUpdateGradOp::gradInIndex()) {
    auto gradInfo = inInfo(ScatterUpdateGradOp::gradInIndex());
    return createDataTensor(
        graph(), gradInfo, plan, axis, 1U, /*broadcasted=*/true, dnai);
  }

  auto indicesInfo = inInfo(ScatterUpdateGradOp::indicesInIndex());
  return createIndicesTensor(graph(), indicesInfo, plan, axis, 1U, true, dnai);
}

InputCreatorType
ScatterUpdateGradOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterUpdateGradOp::gradInIndex() ||
      index == ScatterUpdateGradOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }
  return Opx::getInputCreatorType(index);
}

namespace {
OpxCreator<ScatterOpx> scatterOpxCreator({Onnx::Operators::Scatter_9,
                                          Onnx::Operators::Scatter_11});
OpxCreator<ScatterDataGradOpx>
    scatterDataGradOpxCreator(Onnx::GradOperators::ScatterDataGrad);
OpxCreator<ScatterUpdateGradOpx>
    scatterUpdateGradOpxCreator(Onnx::GradOperators::ScatterUpdateGrad);
} // namespace

} // namespace popx
} // namespace popart
