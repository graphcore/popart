#include <popart/error.hpp>
#include <popart/op/topk.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/topkx.hpp>
#include <popart/popx/opxmanager.hpp>

#include <popops/Zero.hpp>

namespace popart {
namespace popx {

TopKOpx::TopKOpx(Op *op, Devicex *devicex) : BaseSortOpx(op, devicex) {
  verifyOp<TopKOp>(op);
  K = static_cast<unsigned>(dynamic_cast<TopKOp *>(op)->getK());
}

TopKGradOpx::TopKGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
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

  poplar::Tensor dataGrad =
      graph().addVariable(gradIn.elementType(), getGradOutShape());

  poputil::mapTensorLinearly(graph(), dataGrad);

  popops::zero(graph(), dataGrad, prog, debugPrefix("zero"));

  scatterutilx::growScatter(prog, graph(), indices, gradIn, dataGrad, axis);

  setOutTensor(TopKGradOp::gradOutIndex(), dataGrad);
}

void TopKOpx::grow(poplar::program::Sequence &prog) const {

  FullSortResult result = growFullSortResult(prog);
  const auto size       = result.values.dim(axis);

  // TODO T8134  I think this is correct, but should confirm:
  // topKVals[:...:,0,:...:] > topKVals[:...:,1,:...:] etc, where the slice
  // is axis here. That is why I have included the reverse.

  auto topKVals = result.values.slice(size - K, size, axis).reverse(axis);
  auto topKInds = result.indices.slice(size - K, size, axis).reverse(axis);

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
