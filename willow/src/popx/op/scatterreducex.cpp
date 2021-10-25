// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/scatterreduce.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/scatterreducex.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Gather.hpp>
#include <popops/Scatter.hpp>
#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

ScatterReduceOpx::ScatterReduceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex), plan(), axis() {
  verifyOp<ScatterReduceOp>(op, {Onnx::CustomOperators::ScatterReduce});

  auto &srop   = getOp<ScatterReduceOp>();
  axis         = static_cast<size_t>(srop.getAxis());
  auto options = createSlicePlanOptions(SlicePlanUsedFor::UpdateAdd,
                                        srop.getAvailableMemoryProportion());
  plan         = createSlicePlan(graph(),
                         inInfo(srop.dataInIndex()),
                         inInfo(srop.indicesInIndex()),
                         options);

  // We always want the ScatterReduce to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceOpx::grow(snap::program::Sequence &prog) const {
  auto data    = getInTensor(ScatterReduceOp::dataInIndex());
  auto indices = getInTensor(ScatterReduceOp::indicesInIndex());

  auto out = createDataTensor(graph(),
                              outInfo(ScatterReduceOp::outIndex()),
                              plan,
                              axis,
                              getDebugNameAndId("scatterreduceOutput"));

  popops::fill(graph().getPoplarGraph(),
               out.getPoplarTensor(),
               prog.getPoplarSequence(),
               0.0f,
               debugContext("scatterreduceFill"));

  // popops::multiUpdateAdd is roughly:
  //   for i indices:
  //    out[indices[i]] += data[i]
  // but the output must be 2d.  To support inputs with rank > 2:
  //   * permute dims of data and indices and output so that slice axis == 0
  //   * indices are linearized into a 1-d coordinate system
  //   * flatten the remaining dims
  auto target = out.dimRoll(axis);
  data        = data.dimRoll(axis);
  indices     = indices.dimRoll(axis);

  if (indices.rank() < 2) {
    // popops::multiUpdateAdd requires 2-d inputs
    target  = target.expand({1});
    indices = indices.expand({1});
    data    = data.expand({1, 1});
  } else {
    target           = target.flatten();
    target           = target.expand({1});
    data             = data.flatten(1, data.rank());
    auto numDataCols = static_cast<int>(data.dim(1));
    indices = scatterutilx::linearizeIndices(*this, prog, indices, numDataCols);
    data    = data.flatten();
    data    = data.expand({1, 1});
  }

  // Assume indices are non-negative
  indices = indices.reinterpret(poplar::UNSIGNED_INT);

  auto scale = graph().getPoplarGraph().addConstant(
      data.elementType(), {}, 1.0f, debugContext("constOne"));
  graph().getPoplarGraph().setTileMapping(scale, 0);

  popops::multiUpdateAdd(graph().getPoplarGraph(),
                         target.getPoplarTensor(),
                         data.getPoplarTensor(),
                         indices.getPoplarTensor(),
                         scale,
                         {0},
                         {1},
                         prog.getPoplarSequence(),
                         plan,
                         poplar::OptionFlags(),
                         debugContext("scatterAdd"));

  setOutTensor(ScatterReduceOp::outIndex(), out);
}

snap::Tensor
ScatterReduceOpx::createInputTensor(InIndex index,
                                    const poplar::DebugNameAndId &dnai) const {

  if (index != ScatterReduceOp::dataInIndex() &&
      index != ScatterReduceOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  auto dataInfo    = inInfo(ScatterReduceOp::dataInIndex());
  auto indicesInfo = inInfo(ScatterReduceOp::indicesInIndex());

  if (index == ScatterReduceOp::dataInIndex()) {
    return createUpdateTensor(graph(), dataInfo, indicesInfo, plan, axis, dnai);
  }

  return createIndicesTensor(graph(), indicesInfo, plan, axis, dnai);
}

InputCreatorType ScatterReduceOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterReduceOp::dataInIndex() ||
      index == ScatterReduceOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return PopOpx::getInputCreatorType(index);
}

ScatterReduceGradOpx::ScatterReduceGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ScatterReduceGradOp>(
      op, {Onnx::CustomGradOperators::ScatterReduceGradOp});

  auto &srop   = getOp<ScatterReduceGradOp>();
  axis         = static_cast<size_t>(srop.getAxis());
  auto options = createSlicePlanOptions(SlicePlanUsedFor::Slice,
                                        srop.getAvailableMemoryProportion());
  plan         = createSlicePlan(graph(),
                         inInfo(srop.gradInIndex()),
                         inInfo(srop.indicesInIndex()),
                         options);

  // We always want the ScatterReduceGrad to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceGradOpx::grow(snap::program::Sequence &prog) const {
  auto gradIn       = getInTensor(ScatterReduceGradOp::gradInIndex());
  auto indices      = getInTensor(ScatterReduceGradOp::indicesInIndex());
  auto gradOutShape = outInfo(ScatterReduceGradOp::gradOutIndex()).shape();

  auto gradOut =
      scatterutilx::growScatterUpdateGrad(*this,
                                          prog,
                                          graph(),
                                          gradIn,
                                          indices,
                                          gradOutShape,
                                          axis,
                                          plan,
                                          getDebugNameAndId("scatterAddGrad"));

  setOutTensor(ScatterReduceGradOp::gradOutIndex(), gradOut);
}

snap::Tensor ScatterReduceGradOpx::createInputTensor(
    InIndex index,
    const poplar::DebugNameAndId &dnai) const {

  if (index != ScatterReduceGradOp::gradInIndex() &&
      index != ScatterReduceGradOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  if (index == ScatterReduceGradOp::gradInIndex()) {
    return createDataTensor(graph(), inInfo(index), plan, axis, dnai);
  }

  return createIndicesTensor(graph(), inInfo(index), plan, axis, dnai);
}

InputCreatorType
ScatterReduceGradOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterReduceGradOp::gradInIndex() ||
      index == ScatterReduceGradOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }
  return PopOpx::getInputCreatorType(index);
}

namespace {
OpxCreator<ScatterReduceOpx>
    scatterReduceOpxCreator(Onnx::CustomOperators::ScatterReduce);
OpxCreator<ScatterReduceGradOpx>
    scatterReduceGradOpxCreator(Onnx::CustomGradOperators::ScatterReduceGradOp);
} // namespace

} // namespace popx
} // namespace popart
