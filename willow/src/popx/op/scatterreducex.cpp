// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/scatterreduce.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/scatterreducex.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Gather.hpp>
#include <popops/Scatter.hpp>
#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

namespace {
popops::SlicePlan generatePlan(const poplar::Graph &graph,
                               const ScatterReduceOp &op,
                               size_t axis) {

  auto dataInfo    = op.inInfo(op.dataInIndex());
  auto indicesInfo = op.inInfo(op.indicesInIndex());

  size_t numEntries = dataInfo.nelms();
  size_t outputSize = 1;
  size_t numLookups = indicesInfo.nelms();

  return popops::embedding::plan(
      graph, popType(dataInfo), numEntries, outputSize, {numLookups}, {});
}
} // namespace

ScatterReduceOpx::ScatterReduceOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex), plan(), axis() {
  verifyOp<ScatterReduceOp>(op, {Onnx::CustomOperators::ScatterReduce});

  auto &srop = getOp<ScatterReduceOp>();
  axis       = static_cast<size_t>(srop.getAxis());
  plan       = generatePlan(graph(), srop, axis);

  // We always want the ScatterReduce to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceOpx::grow(poplar::program::Sequence &prog) const {
  auto data                 = getInTensor(ScatterReduceOp::dataInIndex());
  auto indices              = getInTensor(ScatterReduceOp::indicesInIndex());
  auto &op                  = getOp<ScatterReduceOp>();
  auto outInfo              = op.outInfo(ScatterReduceOp::outIndex());
  auto shape                = outInfo.shape_szt();
  auto poplarType           = popType(outInfo);
  std::vector<size_t> dims  = {axis};
  std::vector<size_t> sizes = {1};

  auto out = popops::createSliceableTensor(graph(),
                                           poplarType,
                                           shape,
                                           dims,
                                           sizes,
                                           plan,
                                           poplar::OptionFlags(),
                                           debugContext("scatterreduceOutput"));

  popops::fill(graph(), out, prog, 0.0f, debugContext("scatterFill"));

  // popops::multiUpdateAdd is roughly:
  //   for i indices:
  //    out[indices[i]] += data[i]
  // but the output must be 2d.  To support inputs with rank > 2:
  //   * permute dims of both data and output so that slice axis == 0
  //   * flatten the remaining dims
  auto target = out;
  target      = target.dimRoll(static_cast<unsigned>(axis));
  target      = target.flatten(1, target.rank());

  // Same for the data updates, but also need a singleton dim
  data = data.dimRoll(static_cast<unsigned>(axis));
  data = data.flatten(1, data.rank());
  data = data.expand({1});

  if (indices.rank() > 1) {
    indices = indices.dimRoll(static_cast<unsigned>(axis));
    indices = indices.flatten(1, indices.rank());

    // Linearize the indices: map from 2-d indices to 1-d
    int numCols     = static_cast<int>(indices.dim(1));
    auto colIndices = scatterutilx::linspace(graph(),
                                             0,
                                             numCols,
                                             getDebugNameAndId("colIds"),
                                             1,
                                             indices.elementType());

    // numCols * indices + colIndices
    indices           = cloneNcopy(prog, indices, "copyIndices");
    auto numColsConst = graph().addConstant(
        indices.elementType(), {}, numCols, getDebugNameAndId("numCols"));
    graph().setTileMapping(numColsConst, 0);

    popops::mulInPlace(graph(),
                       indices,
                       numColsConst,
                       prog,
                       getDebugNameAndId("numColsMulIndices"));
    popops::addInPlace(graph(),
                       indices,
                       colIndices,
                       prog,
                       getDebugNameAndId("indicesAddColIds"));

    indices = indices.flatten();
    target  = target.flatten();
    target  = target.expand({1});
    data    = data.flatten();
    data    = data.expand({1, 1});
  }

  // Add singleton dim to indices and assume non-negative
  indices = indices.expand({1});
  indices = indices.reinterpret(poplar::UNSIGNED_INT);

  auto scale = graph().addConstant(
      data.elementType(), {}, 1.0f, debugContext("constOne"));
  graph().setTileMapping(scale, 0);

  popops::multiUpdateAdd(graph(),
                         target,
                         data,
                         indices,
                         scale,
                         {0},
                         sizes,
                         prog,
                         plan,
                         poplar::OptionFlags(),
                         debugContext("scatterAdd"));

  setOutTensor(ScatterReduceOp::outIndex(), out);
}

poplar::Tensor
ScatterReduceOpx::createInput(InIndex index,
                              const poplar::DebugNameAndId &dnai) const {

  if (index != ScatterReduceOp::dataInIndex() &&
      index != ScatterReduceOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  std::vector<size_t> dims  = {axis};
  std::vector<size_t> sizes = {1};

  if (index == ScatterReduceOp::dataInIndex()) {
    auto dataInfo        = inInfo(ScatterReduceOp::dataInIndex());
    const auto dataShape = dataInfo.shape_szt();

    return popops::createSliceableTensor(graph(),
                                         popType(dataInfo),
                                         dataShape,
                                         dims,
                                         sizes,
                                         plan,
                                         poplar::OptionFlags(),
                                         dnai);
  }

  auto indicesInfo = inInfo(ScatterReduceOp::indicesInIndex());
  auto indices     = popops::createIndicesTensor(
      graph(), dims, indicesInfo.nelms(), plan, poplar::OptionFlags(), dnai);
  indices = indices.reinterpret(popType(indicesInfo));
  return indices.reshape(indicesInfo.shape_szt());
}

InputCreatorType ScatterReduceOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterReduceOp::dataInIndex() ||
      index == ScatterReduceOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return Opx::getInputCreatorType(index);
}

ScatterReduceGradOpx::ScatterReduceGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<ScatterReduceGradOp>(
      op, {Onnx::CustomGradOperators::ScatterReduceGradOp});

  auto &srop = getOp<ScatterReduceGradOp>();
  axis       = static_cast<size_t>(srop.getAxis());
}

void ScatterReduceGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradIn  = getInTensor(ScatterReduceGradOp::gradInIndex());
  auto indices = getInTensor(ScatterReduceGradOp::indicesInIndex());

  auto gradOut = scatterutilx::growScatterUpdateGrad(
      prog,
      graph(),
      gradIn,
      indices,
      axis,
      getDebugNameAndId("scatterreduceGrad"));

  setOutTensor(ScatterReduceGradOp::gradOutIndex(), gradOut);
}

namespace {
OpxCreator<ScatterReduceOpx>
    scatterReduceOpxCreator(Onnx::CustomOperators::ScatterReduce);
OpxCreator<ScatterReduceGradOpx>
    scatterReduceGradOpxCreator(Onnx::CustomGradOperators::ScatterReduceGradOp);
} // namespace

} // namespace popx
} // namespace popart
