// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popnn/Loss.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Zero.hpp>
#include <poputil/TileMapping.hpp>
#include <popart/op/topk.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/op/topkx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/op/basesortx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensorinfo.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace popart {
namespace popx {
class Devicex;

TopKOpx::TopKOpx(Op *op, Devicex *devicex) : BaseSortOpx(op, devicex) {
  verifyOp<TopKOp>(op);
  K = static_cast<unsigned>(dynamic_cast<TopKOp *>(op)->getK());
}

TopKGradOpx::TopKGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<TopKGradOp>(op, Onnx::GradOperators::TopKGrad);

  auto &topKGradOp = getOp<TopKGradOp>();

  axis        = topKGradOp.getAxis();
  gradOutInfo = topKGradOp.getGradOutInfo();

  const auto options = createSlicePlanOptions(
      SlicePlanUsedFor::Update, topKGradOp.getAvailableMemoryProportion());
  plan = createSlicePlan(graph(),
                         outInfo(topKGradOp.gradOutIndex()),
                         inInfo(topKGradOp.indicesInIndex()),
                         options);
}

void TopKGradOpx::grow(poplar::program::Sequence &prog) const {
  auto gradIn  = getInTensor(TopKGradOp::gradInIndex());
  auto indices = getInTensor(TopKGradOp::indicesInIndex());

  const auto uaxis    = static_cast<unsigned>(axis);
  const auto dataInfo = outInfo(TopKGradOp::gradOutIndex());
  auto dataGrad       = createDataTensor(
      graph(), dataInfo, plan, uaxis, 1U, true, getDebugNameAndId("dataGrad"));

  popops::zero(graph(), dataGrad, prog, debugContext("dataGradFill"));

  poplar::Tensor out = scatterutilx::growScatter(
      *this, prog, graph(), dataGrad, gradIn, indices, gradOutInfo, plan, axis);

  setOutTensor(TopKGradOp::gradOutIndex(), out);
}

poplar::Tensor
TopKGradOpx::createInputTensor(InIndex index,
                               const poplar::DebugNameAndId &dnai) const {
  if (index != TopKGradOp::gradInIndex() &&
      index != TopKGradOp::indicesInIndex()) {
    throw error("TopKGradOpx::createInput : Invalid index = {}", index);
  }

  const auto dataInfo    = outInfo(TopKGradOp::gradOutIndex());
  const auto indicesInfo = inInfo(TopKGradOp::indicesInIndex());
  const auto uaxis       = static_cast<unsigned>(axis);

  if (index == TopKGradOp::indicesInIndex()) {
    return createIndicesTensor(
        graph(), indicesInfo, plan, uaxis, 1U, true, dnai);
  }

  return createUpdateTensor(
      graph(), dataInfo, indicesInfo, plan, uaxis, 1U, true, dnai);
}

InputCreatorType TopKGradOpx::getInputCreatorType(InIndex index) const {
  if (index == TopKGradOp::gradInIndex() ||
      index == TopKGradOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return Opx::getInputCreatorType(index);
}

void TopKOpx::grow(poplar::program::Sequence &prog) const {
  auto negateTensor = [&](auto &x) {
    return popops::map(graph(),
                       popops::expr::UnaryOpType::NEGATE,
                       x,
                       prog,
                       debugContext("neg"));
  };

  // Input shape, e.g. for rank = 4, axis = 2:
  //   [a0, a1, a2, a3]
  // Output shape:
  //   [a0, a1, K,  a3]
  auto input = getInTensor(TopKOp::getInIndex());

  auto &topk = getOp<TopKOp>();

  if (!topk.getLargest()) {
    input = negateTensor(input);
  }

  const auto lastDim = input.rank() - 1;
  // Poplibs topk requires input with rank = 2, axis = 1
  // Reshape input to:
  //   [a0*a1*a3, a2]
  if (axis != lastDim) {
    input = input.dimShufflePartial({axis, lastDim}, {lastDim, axis});
  }

  const auto dim1Elememts = input.dim(lastDim);
  const auto dim0Elems    = input.numElements() / dim1Elememts;
  input                   = input.reshape({dim0Elems, dim1Elememts});

  // Add variable to store indices
  auto indsShape = input.shape();
  indsShape[1]   = K;
  auto topKInds  = graph().addVariable(
      poplar::UNSIGNED_INT, indsShape, debugContext("topKInds"));
  poputil::mapTensorLinearly(graph(), topKInds);

  auto topKVals = popnn::topK(graph(),
                              input,
                              topKInds,
                              K,
                              topk.getSorted(),
                              prog,
                              debugContext("topK"));
  if (!topk.getLargest()) {
    topKVals = negateTensor(topKVals);
  }

  // Reverse the dimshuffling and reshaping of the input and indices tensors
  const auto valsShape = outShape(TopKOp::getValuesOutIndex());
  std::vector<size_t> valsShape_t(valsShape.begin(), valsShape.end());
  std::swap(valsShape_t[axis], valsShape_t[lastDim]);

  // of shape: [a0, a1, a3, K]
  topKVals = topKVals.reshape(valsShape_t);

  topKInds = topKInds.reinterpret(poplar::INT);
  topKInds = topKInds.reshape(valsShape_t);

  // of shape [a0, a1, K, a3]
  if (axis != lastDim) {
    topKVals = topKVals.dimShufflePartial({axis, lastDim}, {lastDim, axis});
    topKInds = topKInds.dimShufflePartial({axis, lastDim}, {lastDim, axis});
  }

  setOutTensor(TopKOp::getValuesOutIndex(), topKVals);
  setOutTensor(TopKOp::getIndicesOutIndex(), topKInds);
}

namespace {
OpxCreator<TopKOpx> TopKOpxCreator({Onnx::Operators::TopK_1,
                                    Onnx::Operators::TopK_10,
                                    Onnx::Operators::TopK_11});
OpxCreator<TopKGradOpx> topkGradOpxCreator(Onnx::GradOperators::TopKGrad);
} // namespace

} // namespace popx
} // namespace popart
