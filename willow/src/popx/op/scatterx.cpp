// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/scatter.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/scatterx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/util.hpp>

#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Gather.hpp>
#include <popops/Scatter.hpp>
#include <poputil/TileMapping.hpp>

namespace popart {
namespace popx {

namespace {

snap::Tensor
concat(const std::vector<snap::Tensor> &ts, unsigned d, snap::Graph &graph) {
  std::vector<poplar::Tensor> tsP;
  tsP.reserve(ts.size());
  for (auto t : ts) {
    tsP.push_back(t.getPoplarTensor());
  }

  return snap::Tensor{poplar::concat(tsP, d), graph};
}

} // unnamed namespace

ScatterOpx::ScatterOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex), plan(), axis() {
  verifyOp<ScatterOp>(
      op, {Onnx::Operators::Scatter_9, Onnx::Operators::Scatter_11});
  auto &sop    = getOp<ScatterOp>();
  axis         = sop.getAxis();
  auto options = createSlicePlanOptions(SlicePlanUsedFor::Update);
  plan         = createSlicePlan(graph(),
                         sop.inInfo(sop.dataInIndex()),
                         sop.inInfo(sop.indicesInIndex()),
                         options);

  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterOpx::grow(poplar::program::Sequence &prog) const {
  auto dataInput = getInTensor(ScatterOp::dataInIndex());
  auto indices   = getInTensor(ScatterOp::indicesInIndex());
  auto values    = getInTensor(ScatterOp::updatesInIndex());
  auto dataInfo  = inInfo(ScatterOp::dataInIndex());
  auto uaxis     = static_cast<unsigned>(axis);

  auto sliceable = popops::createSliceableTensor(graph().getPoplarGraph(),
                                                 dataInput.elementType(),
                                                 {dataInput.numElements(), 1},
                                                 {0},
                                                 {1},
                                                 plan,
                                                 poplar::OptionFlags(),
                                                 debugContext("scatterOuput"));

  auto out =
      alignToAxis(snap::Tensor{sliceable, graph()}, dataInfo.shape(), uaxis);

  prog.add(poplar::program::Copy(dataInput.getPoplarTensor(),
                                 out.getPoplarTensor(),
                                 false,
                                 debugContext("copyToScatter")));
  indices = indices.dimRoll(uaxis);
  values  = values.dimRoll(uaxis);

  if (indices.rank() < 2) {
    // popops::multiUpdate requires 2-d inputs
    indices = indices.expand({1});
    values  = values.expand({1, 1});
  } else {
    auto numDataCols = dataInfo.nelms() / dataInfo.shape().at(uaxis);
    indices = scatterutilx::linearizeIndices(*this, prog, indices, numDataCols);
    values  = values.flatten();
    values  = values.expand({1, 1});
  }

  // Assume indices are non-negative
  indices = indices.reinterpret(poplar::UNSIGNED_INT);

  popops::multiUpdate(graph().getPoplarGraph(),
                      sliceable,
                      values.getPoplarTensor(),
                      indices.getPoplarTensor(),
                      {0},
                      {1},
                      prog,
                      plan,
                      poplar::OptionFlags(),
                      debugContext("scatter"));

  setOutTensor(ScatterOp::outIndex(), out);
}

snap::Tensor
ScatterOpx::createInputTensor(InIndex index,
                              const poplar::DebugNameAndId &dnai) const {
  if (index != ScatterOp::indicesInIndex() &&
      index != ScatterOp::updatesInIndex()) {
    throw error("ScatterOpx::createInput : Invalid index = {}", index);
  }

  auto dataInfo             = inInfo(ScatterOp::dataInIndex());
  auto indicesInfo          = inInfo(ScatterOp::indicesInIndex());
  auto numEntries           = static_cast<size_t>(dataInfo.nelms());
  auto numLookups           = static_cast<size_t>(indicesInfo.nelms());
  size_t outputSize         = 1;
  std::vector<size_t> dims  = {0};
  std::vector<size_t> sizes = {outputSize};
  auto uaxis                = static_cast<unsigned>(axis);

  if (index == ScatterOp::indicesInIndex()) {
    auto indices = popops::createIndicesTensor(graph().getPoplarGraph(),
                                               dims,
                                               numLookups,
                                               plan,
                                               poplar::OptionFlags(),
                                               dnai);
    indices      = indices.reinterpret(popType(indicesInfo));
    return alignToAxis(
        snap::Tensor{indices, graph()}, indicesInfo.shape(), uaxis);
  }

  auto updatesInfo = inInfo(ScatterOp::updatesInIndex());
  auto updates     = popops::createSliceTensor(graph().getPoplarGraph(),
                                           popType(updatesInfo),
                                           {numEntries, outputSize},
                                           dims,
                                           sizes,
                                           numLookups,
                                           plan,
                                           poplar::OptionFlags(),
                                           dnai);
  return alignToAxis(
      snap::Tensor{updates, graph()}, updatesInfo.shape(), uaxis);
}

InputCreatorType ScatterOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterOp::indicesInIndex() ||
      index == ScatterOp::updatesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return PopOpx::getInputCreatorType(index);
}

ScatterDataGradOpx::ScatterDataGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ScatterDataGradOp>(op, Onnx::GradOperators::ScatterDataGrad);

  axis = dynamic_cast<ScatterDataGradOp *>(op)->getAxis();
}

void ScatterDataGradOpx::grow(poplar::program::Sequence &prog) const {
  auto data = cloneNcopy(prog, getInTensor(ScatterDataGradOp::gradInIndex()));
  auto indices = getInTensor(ScatterDataGradOp::indicesInIndex());
  auto update  = snap::Tensor{
      graph().getPoplarGraph().addConstant(
          data.elementType(), indices.shape(), 0, debugContext("zeros")),
      graph()};
  poputil::mapTensorLinearly(graph().getPoplarGraph(),
                             update.getPoplarTensor());

  // Build the implicit index coordinates
  //
  // popops::scatter requires the indices to be complete coordinates into the
  // data tensor, but ONNX scatter only provides an axis and a scalar index.
  std::vector<snap::Tensor> indices_mapped(indices.rank());
  for (int i = 0; i < indices_mapped.size(); ++i) {
    auto t = scatterutilx::linspace(graph(),
                                    0,
                                    static_cast<int>(indices.dim(i)),
                                    getDebugNameAndId("linspace"));

    // Match the rank of indices
    t = scatterutilx::matchRank(indices, t, i);

    // Match the shape of indices
    indices_mapped[i] = scatterutilx::broadcastShape(indices, t);
  }

  // Replace the axis indices with the user provided indices
  indices_mapped[axis] = indices;

  // Add a degenerate dimension for concatenation
  for (auto &index : indices_mapped) {
    index = index.expand({index.rank()});
  }

  std::vector<unsigned> update_window_dims(indices_mapped.size());
  std::iota(update_window_dims.begin(), update_window_dims.end(), 0);

  std::vector<std::size_t> inserted_window_dims(indices_mapped.size());
  std::iota(inserted_window_dims.begin(), inserted_window_dims.end(), 0);

  std::vector<unsigned> scatter_dims_to_op(indices_mapped.size());
  std::iota(scatter_dims_to_op.begin(), scatter_dims_to_op.end(), 0);

  // Concat the indices on the degenerate dimension
  indices = concat(indices_mapped, indices_mapped.size(), graph());

  // Scatter the zeros into data
  popops::scatter(graph().getPoplarGraph(),
                  data.getPoplarTensor(),
                  indices.getPoplarTensor(),
                  update.getPoplarTensor(),
                  indices.rank() - 1,
                  update_window_dims,
                  inserted_window_dims,
                  scatter_dims_to_op,
                  prog,
                  debugContext("scatter"));

  setOutTensor(ScatterDataGradOp::gradOutIndex(), data);
}

ScatterUpdateGradOpx::ScatterUpdateGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ScatterUpdateGradOp>(op, Onnx::GradOperators::ScatterUpdateGrad);

  axis = dynamic_cast<ScatterUpdateGradOp *>(op)->getAxis();
}

void ScatterUpdateGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradIn  = getInTensor(ScatterUpdateGradOp::gradInIndex());
  auto indices = getInTensor(ScatterDataGradOp::indicesInIndex());

  auto gradOut = scatterutilx::growScatterUpdateGrad(
      prog,
      graph(),
      gradIn,
      indices,
      axis,
      getDebugNameAndId("scatter_update_grad"));

  setOutTensor(ScatterUpdateGradOp::gradOutIndex(), gradOut);
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
