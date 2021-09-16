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
                         srop.inInfo(srop.dataInIndex()),
                         srop.inInfo(srop.indicesInIndex()),
                         options);

  // We always want the ScatterReduce to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceOpx::grow(poplar::program::Sequence &prog) const {
  auto data                 = getInTensor(ScatterReduceOp::dataInIndex());
  auto indices              = getInTensor(ScatterReduceOp::indicesInIndex());
  auto &op                  = getOp<ScatterReduceOp>();
  auto outInfo              = op.outInfo(ScatterReduceOp::outIndex());
  auto poplarType           = popType(outInfo);
  std::vector<size_t> dims  = {0};
  std::vector<size_t> sizes = {1};

  auto target =
      popops::createSliceableTensor(graph().getPoplarGraph(),
                                    poplarType,
                                    {static_cast<size_t>(outInfo.nelms()), 1},
                                    dims,
                                    sizes,
                                    plan,
                                    poplar::OptionFlags(),
                                    debugContext("scatterreduceOutput"));

  popops::fill(graph().getPoplarGraph(),
               target,
               prog,
               0.0f,
               debugContext("scatterFill"));

  // popops::multiUpdateAdd is roughly:
  //   for i indices:
  //    out[indices[i]] += data[i]
  // but the output must be 2d.  To support inputs with rank > 2:
  //   * permute dims of data and indices and output so that slice axis == 0
  //   * indices are linearized into a 1-d coordinate system
  //   * flatten the remaining dims
  data    = data.dimRoll(axis);
  indices = indices.dimRoll(axis);

  if (indices.rank() < 2) {
    // popops::multiUpdateAdd requires 2-d inputs
    indices = indices.expand({1});
    data    = data.expand({1, 1});
  } else {
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
                         target,
                         data.getPoplarTensor(),
                         indices.getPoplarTensor(),
                         scale,
                         {0},
                         sizes,
                         prog,
                         plan,
                         poplar::OptionFlags(),
                         debugContext("scatterAdd"));

  auto out = alignToAxis(snap::Tensor{target, graph()}, outInfo.shape(), axis);
  setOutTensor(ScatterReduceOp::outIndex(), out);
}

snap::Tensor
ScatterReduceOpx::createInputTensor(InIndex index,
                                    const poplar::DebugNameAndId &dnai) const {

  if (index != ScatterReduceOp::dataInIndex() &&
      index != ScatterReduceOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  auto dataInfo             = inInfo(ScatterReduceOp::dataInIndex());
  auto numEntries           = static_cast<size_t>(dataInfo.nelms());
  auto indicesInfo          = inInfo(ScatterReduceOp::indicesInIndex());
  auto numLookups           = static_cast<size_t>(indicesInfo.nelms());
  size_t outputSize         = 1;
  std::vector<size_t> dims  = {0};
  std::vector<size_t> sizes = {outputSize};

  if (index == ScatterReduceOp::dataInIndex()) {
    auto data = popops::createSliceTensor(graph().getPoplarGraph(),
                                          popType(dataInfo),
                                          {numEntries, outputSize},
                                          dims,
                                          sizes,
                                          numLookups,
                                          plan,
                                          poplar::OptionFlags(),
                                          dnai);
    return alignToAxis(snap::Tensor{data, graph()}, dataInfo.shape(), axis);
  }

  auto indices = popops::createIndicesTensor(graph().getPoplarGraph(),
                                             dims,
                                             numLookups,
                                             plan,
                                             poplar::OptionFlags(),
                                             dnai);
  indices      = indices.reinterpret(popType(indicesInfo));
  return alignToAxis(snap::Tensor{indices, graph()}, indicesInfo.shape(), axis);
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
                         srop.inInfo(srop.gradInIndex()),
                         srop.inInfo(srop.indicesInIndex()),
                         options);

  // We always want the ScatterReduceGrad to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradIn  = getInTensor(ScatterReduceGradOp::gradInIndex());
  auto indices = getInTensor(ScatterReduceGradOp::indicesInIndex());

  // Place the gather axis at the front.
  gradIn  = gradIn.dimRoll(axis);
  indices = indices.dimRoll(axis);

  if (indices.rank() < 2) {
    indices = indices.expand({1});
    gradIn  = gradIn.expand({1});
  } else {
    auto numCols = indices.numElements() / indices.shape().at(0);
    indices = scatterutilx::linearizeIndices(*this, prog, indices, numCols);
    gradIn  = gradIn.flatten();
    gradIn  = gradIn.expand({1});
  }

  // Assume indices are non-negative
  indices = indices.reinterpret(poplar::UNSIGNED_INT);

  auto result = popops::multiSlice(graph().getPoplarGraph(),
                                   gradIn.getPoplarTensor(),
                                   indices.getPoplarTensor(),
                                   {0},
                                   {1},
                                   prog,
                                   plan,
                                   poplar::OptionFlags(),
                                   debugContext("scatterAddGrad"));

  auto shape   = outInfo(ScatterReduceGradOp::gradOutIndex()).shape();
  auto gradOut = alignToAxis(snap::Tensor(result, graph()), shape, axis);
  setOutTensor(ScatterReduceGradOp::gradOutIndex(), gradOut);
}

snap::Tensor ScatterReduceGradOpx::createInputTensor(
    InIndex index,
    const poplar::DebugNameAndId &dnai) const {

  if (index != ScatterReduceGradOp::gradInIndex() &&
      index != ScatterReduceGradOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  std::vector<size_t> dims  = {0};
  std::vector<size_t> sizes = {1};

  if (index == ScatterReduceGradOp::gradInIndex()) {
    auto gradInfo   = inInfo(ScatterReduceGradOp::gradInIndex());
    auto numEntries = static_cast<size_t>(gradInfo.nelms());
    auto grad       = popops::createSliceableTensor(graph().getPoplarGraph(),
                                              popType(gradInfo),
                                              {numEntries, 1},
                                              dims,
                                              sizes,
                                              plan,
                                              poplar::OptionFlags(),
                                              dnai);

    return alignToAxis(snap::Tensor{grad, graph()}, gradInfo.shape(), axis);
  }

  auto indicesInfo = inInfo(ScatterReduceGradOp::indicesInIndex());
  auto numLookups  = static_cast<size_t>(indicesInfo.nelms());
  auto indices     = popops::createIndicesTensor(graph().getPoplarGraph(),
                                             dims,
                                             numLookups,
                                             plan,
                                             poplar::OptionFlags(),
                                             dnai);
  indices          = indices.reinterpret(popType(indicesInfo));
  return alignToAxis(snap::Tensor{indices, graph()}, indicesInfo.shape(), axis);
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
