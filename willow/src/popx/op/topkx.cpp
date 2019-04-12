#include <poponnx/error.hpp>
#include <poponnx/op/topk.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/topkx.hpp>
#include <poponnx/popx/opxmanager.hpp>

namespace poponnx {
namespace popx {

TopKOpx::TopKOpx(Op *op, Devicex *devicex) : BaseSortOpx(op, devicex) {
  verifyOp<TopKOp>(op);
  K = static_cast<unsigned>(dynamic_cast<TopKOp *>(op)->getK());
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
OpxCreator<TopKOpx> TopKOpxCreator(Onnx::Operators::TopK_1);
} // namespace

} // namespace popx
} // namespace poponnx
