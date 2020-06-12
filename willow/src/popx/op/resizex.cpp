#include <cmath>

#include <popops/ElementWise.hpp>
#include <popops/Zero.hpp>

#include <popart/logging.hpp>
#include <popart/op/resize.hpp>
#include <popart/popx/op/resizex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

ResizeOpx::ResizeOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ResizeOp>(op);
}

void ResizeOpx::grow(poplar::program::Sequence &prog) const {
  auto &resizeOp = getOp<ResizeOp>();
  auto outShape  = resizeOp.outShape(ResizeOp::getOutIndex());

  auto input  = getInTensor(ResizeOp::getInIndex());
  auto result = cloneNcopy(prog, input);
  for (int i = 0; i < input.rank(); i++) {
    if (result.shape().at(i) != outShape.at(i)) {
      result = resize_nearest(result, i, outShape.at(i));
    }
  }

  setOutTensor(ResizeOp::getOutIndex(), result);
}

namespace {

std::vector<poplar::Tensor> split(poplar::Tensor &input, int dim) {
  std::vector<poplar::Tensor> result;
  for (int i = 0; i < input.dim(dim); i++) {
    result.push_back(input.slice(i, i + 1, dim));
  }
  return result;
}

} // namespace

poplar::Tensor
ResizeOpx::resize_nearest(poplar::Tensor &input, int dim, int size) const {
  auto slices = split(input, dim);

  std::vector<poplar::Tensor> toConcat;
  float k = static_cast<float>(input.dim(dim)) / static_cast<float>(size);
  for (int i = 0; i < size; i++) {
    int idx = std::floor(i * k);
    toConcat.push_back(slices.at(idx));
  }

  return poplar::concat(toConcat, dim);
}

namespace {
OpxCreator<ResizeOpx> resizeOpxCreator(Onnx::Operators::Resize_10);
} // namespace

} // namespace popx
} // namespace popart
