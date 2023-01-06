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

namespace {
poplar::Tensor scatter(const popart::popx::Opx &opx,
                       poplar::program::Sequence &prog,
                       poplar::Graph &graph,
                       const poplar::Tensor &data,
                       const poplar::Tensor &updates,
                       const poplar::Tensor &indices,
                       const popart::TensorInfo &dataInfo,
                       const popops::SlicePlan &plan,
                       int64_t axis) {
  auto uaxis = static_cast<unsigned>(axis);
  auto out   = popart::popx::createDataTensor(graph,
                                            dataInfo,
                                            plan,
                                            uaxis,
                                            1U,
                                            /* broadcasted= */ true,
                                            opx.getDebugNameAndId(""));

  prog.add(poplar::program::Copy(
      data, out, false, opx.debugContext("copyToScatter")));

  auto data2d    = out.dimRoll(uaxis);
  auto updates2d = updates.dimRoll(uaxis);
  auto indices2d = indices.dimRoll(uaxis);

  if (indices2d.rank() < 2) {
    // popops::multiUpdate requires 2-d inputs
    data2d    = data2d.expand({1});
    updates2d = updates2d.expand({1, 1});
    indices2d = indices2d.expand({1});
  } else {
    data2d = data2d.flatten();
    data2d = data2d.expand({1});

    auto numDataCols = dataInfo.nelms() / dataInfo.shape().at(uaxis);
    indices2d        = popart::popx::scatterutilx::linearizeIndices(
        opx, prog, indices2d, numDataCols, 1U);

    updates2d = updates2d.flatten();
    updates2d = updates2d.expand({1, 1});
  }

  // Assume indices are non-negative
  indices2d = indices2d.reinterpret(poplar::UNSIGNED_INT);

  popops::multiUpdate(graph,
                      data2d,
                      updates2d,
                      indices2d,
                      {0},
                      {1},
                      prog,
                      plan,
                      poplar::OptionFlags(),
                      opx.debugContext("scatter"));

  return out;
}
} // namespace

namespace popart {
namespace popx {

ScatterOpx::ScatterOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex), plan(), axis() {
  verifyOp<ScatterOp>(
      op, {Onnx::Operators::Scatter_9, Onnx::Operators::Scatter_11});
  auto &sop    = getOp<ScatterOp>();
  axis         = sop.getAxis();
  auto options = createSlicePlanOptions(SlicePlanUsedFor::Update,
                                        sop.getAvailableMemoryProportion());
  plan         = createSlicePlan(graph(),
                         inInfo(sop.dataInIndex()),
                         inInfo(sop.indicesInIndex()),
                         options);

  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterOpx::grow(poplar::program::Sequence &prog) const {
  auto scatterOut = scatter(*this,
                            prog,
                            graph(),
                            getInTensor(ScatterOp::dataInIndex()),
                            getInTensor(ScatterOp::updatesInIndex()),
                            getInTensor(ScatterOp::indicesInIndex()),
                            inInfo(ScatterOp::dataInIndex()),
                            plan,
                            static_cast<unsigned>(axis));

  setOutTensor(ScatterOp::outIndex(), scatterOut);
}

poplar::Tensor
ScatterOpx::createInput(InIndex index,
                        const poplar::DebugNameAndId &dnai) const {
  if (index != ScatterOp::indicesInIndex() &&
      index != ScatterOp::updatesInIndex()) {
    throw error("ScatterOpx::createInput : Invalid index = {}", index);
  }

  auto indicesInfo = inInfo(ScatterOp::indicesInIndex());
  auto uaxis       = static_cast<unsigned>(axis);

  if (index == ScatterOp::indicesInIndex()) {
    return createIndicesTensor(
        graph(), indicesInfo, plan, uaxis, 1U, true, dnai);
  }

  return createUpdateTensor(graph(),
                            inInfo(ScatterOp::dataInIndex()),
                            indicesInfo,
                            plan,
                            uaxis,
                            1U,
                            /*broadcasted=*/true,
                            dnai);
}

InputCreatorType ScatterOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterOp::indicesInIndex() ||
      index == ScatterOp::updatesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return Opx::getInputCreatorType(index);
}

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

  auto gradOut = scatter(*this,
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
