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

namespace {
poplar::Tensor linearizeIndices(const PopOpx &opx,
                                poplar::program::Sequence &prog,
                                poplar::Tensor indices) {
  // Linearize the indices: map from 2-d indices to 1-d
  auto result     = indices.flatten(1, indices.rank());
  int numCols     = static_cast<int>(result.dim(1));
  auto colIndices = scatterutilx::linspace(opx.graph(),
                                           0,
                                           numCols,
                                           opx.getDebugNameAndId("colIds"),
                                           1,
                                           result.elementType());

  // numCols * indices + colIndices
  result =
      opx.cloneNcopy(prog, snap::Tensor{result, opx.graph()}, "copyIndices")
          .getPoplarTensor();
  auto numColsConst = opx.graph().getPoplarGraph().addConstant(
      result.elementType(), {}, numCols, opx.getDebugNameAndId("numCols"));
  opx.graph().getPoplarGraph().setTileMapping(numColsConst, 0);

  popops::mulInPlace(opx.graph().getPoplarGraph(),
                     result,
                     numColsConst,
                     prog,
                     opx.getDebugNameAndId("numColsMulIndices"));
  popops::addInPlace(opx.graph().getPoplarGraph(),
                     result,
                     colIndices.getPoplarTensor(),
                     prog,
                     opx.getDebugNameAndId("indicesAddColIds"));

  result = result.flatten();
  result = result.expand({1});
  return result;
}
} // namespace

ScatterReduceOpx::ScatterReduceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex), plan(), axis() {
  verifyOp<ScatterReduceOp>(op, {Onnx::CustomOperators::ScatterReduce});

  auto &srop = getOp<ScatterReduceOp>();
  axis       = static_cast<size_t>(srop.getAxis());
  plan       = createSlicePlan(graph(),
                         srop.inInfo(srop.dataInIndex()),
                         srop.inInfo(srop.indicesInIndex()),
                         srop.getAvailableMemoryProportion());

  // We always want the ScatterReduce to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceOpx::grow(poplar::program::Sequence &prog) const {
  auto data = getInTensor(ScatterReduceOp::dataInIndex()).getPoplarTensor();
  auto indices =
      getInTensor(ScatterReduceOp::indicesInIndex()).getPoplarTensor();
  auto &op                  = getOp<ScatterReduceOp>();
  auto outInfo              = op.outInfo(ScatterReduceOp::outIndex());
  auto shape                = outInfo.shape_szt();
  auto poplarType           = popType(outInfo);
  std::vector<size_t> dims  = {axis};
  std::vector<size_t> sizes = {1};

  auto out = popops::createSliceableTensor(graph().getPoplarGraph(),
                                           poplarType,
                                           shape,
                                           dims,
                                           sizes,
                                           plan,
                                           poplar::OptionFlags(),
                                           debugContext("scatterreduceOutput"));

  popops::fill(
      graph().getPoplarGraph(), out, prog, 0.0f, debugContext("scatterFill"));

  // popops::multiUpdateAdd is roughly:
  //   for i indices:
  //    out[indices[i]] += data[i]
  // but the output must be 2d.  To support inputs with rank > 2:
  //   * permute dims of data and indices and output so that slice axis == 0
  //   * indices are linearized into a 1-d coordinate system
  //   * flatten the remaining dims
  auto target = out;
  target      = target.dimRoll(static_cast<unsigned>(axis));
  data        = data.dimRoll(static_cast<unsigned>(axis));
  indices     = indices.dimRoll(static_cast<unsigned>(axis));

  if (indices.rank() < 2) {
    // popops::multiUpdateAdd requires 2-d inputs
    indices = indices.expand({1});
    target  = target.expand({1});
    data    = data.expand({1, 1});
  } else {
    indices = linearizeIndices(*this, prog, indices);
    target  = target.flatten();
    target  = target.expand({1});
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
                         data,
                         indices,
                         scale,
                         {0},
                         sizes,
                         prog,
                         plan,
                         poplar::OptionFlags(),
                         debugContext("scatterAdd"));

  setOutTensor(ScatterReduceOp::outIndex(), snap::Tensor{out, graph()});
}

snap::Tensor
ScatterReduceOpx::createInputTensor(InIndex index,
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

    return snap::Tensor{popops::createSliceableTensor(graph().getPoplarGraph(),
                                                      popType(dataInfo),
                                                      dataShape,
                                                      dims,
                                                      sizes,
                                                      plan,
                                                      poplar::OptionFlags(),
                                                      dnai),
                        graph()};
  }

  auto indicesInfo = inInfo(ScatterReduceOp::indicesInIndex());
  auto indices     = popops::createIndicesTensor(graph().getPoplarGraph(),
                                             dims,
                                             indicesInfo.nelms(),
                                             plan,
                                             poplar::OptionFlags(),
                                             dnai);
  indices          = indices.reinterpret(popType(indicesInfo));
  return snap::Tensor{indices.reshape(indicesInfo.shape_szt()), graph()};
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

  auto &srop = getOp<ScatterReduceGradOp>();
  axis       = static_cast<size_t>(srop.getAxis());
  plan       = createSlicePlan(graph(),
                         srop.inInfo(srop.gradInIndex()),
                         srop.inInfo(srop.indicesInIndex()),
                         srop.getAvailableMemoryProportion());

  // We always want the ScatterReduceGrad to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradIn =
      getInTensor(ScatterReduceGradOp::gradInIndex()).getPoplarTensor();
  auto indices =
      getInTensor(ScatterReduceGradOp::indicesInIndex()).getPoplarTensor();

  // Place the gather axis at the front.
  gradIn  = gradIn.dimRoll(static_cast<unsigned>(axis));
  indices = indices.dimRoll(static_cast<unsigned>(axis));

  // Store the shape for later.
  auto tmp_shape = indices.shape();

  if (indices.rank() < 2) {
    indices = indices.expand({1});
    gradIn  = gradIn.expand({1});
  } else {
    indices = linearizeIndices(*this, prog, indices);
    gradIn  = gradIn.flatten();
    gradIn  = gradIn.expand({1});
  }

  // Assume indices are non-negative
  indices = indices.reinterpret(poplar::UNSIGNED_INT);

  auto result = popops::multiSlice(graph().getPoplarGraph(),
                                   gradIn,
                                   indices,
                                   {0},
                                   {1},
                                   prog,
                                   plan,
                                   poplar::OptionFlags(),
                                   debugContext("scatterAddGrad"));

  // Reshape the result to "unflatten" the other dimensions.
  result = result.reshape(tmp_shape);
  // Put the gather axis dimension back in the right place.
  result = result.dimRoll(0, static_cast<unsigned>(axis));

  // Reshape into the expected output shape.
  const auto outputShape =
      outInfo(ScatterReduceGradOp::gradOutIndex()).shape_szt();
  result = result.reshape(outputShape);

  setOutTensor(ScatterReduceGradOp::gradOutIndex(),
               snap::Tensor{result, graph()});
}

snap::Tensor ScatterReduceGradOpx::createInputTensor(
    InIndex index,
    const poplar::DebugNameAndId &dnai) const {

  if (index != ScatterReduceGradOp::gradInIndex() &&
      index != ScatterReduceGradOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  std::vector<size_t> dims  = {axis};
  std::vector<size_t> sizes = {1};

  if (index == ScatterReduceGradOp::gradInIndex()) {
    auto dataInfo        = inInfo(ScatterReduceGradOp::gradInIndex());
    const auto dataShape = dataInfo.shape_szt();

    return snap::Tensor{popops::createSliceableTensor(graph().getPoplarGraph(),
                                                      popType(dataInfo),
                                                      dataShape,
                                                      dims,
                                                      sizes,
                                                      plan,
                                                      poplar::OptionFlags(),
                                                      dnai),
                        graph()};
  }

  auto indicesInfo = inInfo(ScatterReduceGradOp::indicesInIndex());
  auto indices     = popops::createIndicesTensor(graph().getPoplarGraph(),
                                             dims,
                                             indicesInfo.nelms(),
                                             plan,
                                             poplar::OptionFlags(),
                                             dnai);
  indices          = indices.reinterpret(popType(indicesInfo));
  return snap::Tensor{indices.reshape(indicesInfo.shape_szt()), graph()};
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
