// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/error.hpp>
#include <popart/op/topk.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/topkx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popnn/Loss.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

namespace popart {
namespace popx {

TopKOpx::TopKOpx(Op *op, Devicex *devicex) : BaseSortOpx(op, devicex) {
  verifyOp<TopKOp>(op);
  K = static_cast<unsigned>(dynamic_cast<TopKOp *>(op)->getK());
}

TopKGradOpx::TopKGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<TopKGradOp>(op, Onnx::GradOperators::TopKGrad);
  axis        = dynamic_cast<TopKGradOp *>(op)->getAxis();
  gradOutInfo = dynamic_cast<TopKGradOp *>(op)->getGradOutInfo();
  gradOutShape.reserve(gradOutInfo.rank());
  for (auto &x : gradOutInfo.shape()) {
    gradOutShape.push_back(static_cast<size_t>(x));
  }
}

const std::vector<size_t> &TopKGradOpx::getGradOutShape() const {
  return gradOutShape;
}

void TopKGradOpx::grow(poplar::program::Sequence &prog) const {
  auto indices = getInTensor(TopKGradOp::indicesInIndex());

  auto gradIn = getInTensor(TopKGradOp::gradInIndex());

  poplar::Tensor dataGrad = graph().getPoplarGraph().addVariable(
      gradIn.elementType(), getGradOutShape(), debugContext("dataGrad"));

  poputil::mapTensorLinearly(graph().getPoplarGraph(), dataGrad);

  popops::zero(graph().getPoplarGraph(), dataGrad, prog, debugContext("zero"));

  scatterutilx::growScatter(prog,
                            graph(),
                            indices,
                            gradIn,
                            dataGrad,
                            axis,
                            getDebugNameAndId("scatter"));

  setOutTensor(TopKGradOp::gradOutIndex(), dataGrad);
}

void TopKOpx::grow(poplar::program::Sequence &prog) const {
  auto negateTensor = [&](auto &x) {
    return popops::map(graph().getPoplarGraph(),
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

  auto lastDim = input.rank() - 1;
  // Poplibs topk requires input with rank = 2, axis = 1
  // Reshape input to:
  //   [a0*a1*a3, a2]
  if (axis != lastDim) {
    input = input.dimShufflePartial({axis, lastDim}, {lastDim, axis});
  }

  auto dim1Elememts = input.dim(lastDim);
  auto dim0Elems    = input.numElements() / dim1Elememts;
  input             = input.reshape({dim0Elems, dim1Elememts});

  // Add variable to store indices
  auto indsShape = input.shape();
  indsShape[1]   = K;
  auto topKInds  = graph().getPoplarGraph().addVariable(
      poplar::UNSIGNED_INT, indsShape, debugContext("topKInds"));
  poputil::mapTensorLinearly(graph().getPoplarGraph(), topKInds);

  auto topKVals = popnn::topK(graph().getPoplarGraph(),
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
  auto valsShape = outShape(TopKOp::getValuesOutIndex());
  std::vector<size_t> valsShape_t(valsShape.begin(), valsShape.end());
  std::swap(valsShape_t[axis], valsShape_t[lastDim]);

  // of shape: [a0, a1, a3, K]
  topKVals = topKVals.reshape(valsShape_t);
  topKInds = topKInds.reshape(valsShape_t);

  // of shape [a0, a1, K, a3]
  if (axis != lastDim) {
    topKVals = topKVals.dimShufflePartial({axis, lastDim}, {lastDim, axis});
  }
  if (axis != lastDim) {
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
