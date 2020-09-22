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
      result =
          resize_nearest(result, i, outShape.at(i), resizeOp.getScales().at(i));
    }
  }

  setOutTensor(ResizeOp::getOutIndex(), result);
}

namespace {

std::vector<poplar::Tensor> split(const poplar::Tensor &input, int dim) {
  std::vector<poplar::Tensor> result;
  for (int i = 0; i < input.dim(dim); i++) {
    result.push_back(input.slice(i, i + 1, dim));
  }
  return result;
}

} // namespace

poplar::Tensor ResizeOpx::resize_nearest(poplar::Tensor &input,
                                         int dim,
                                         int64_t size,
                                         float scale) const {
  auto slices = split(input, dim);

  std::vector<poplar::Tensor> toConcat;
  for (int i = 0; i < size; i++) {
    int idx = static_cast<int>(std::floor(static_cast<float>(i) / scale));
    toConcat.push_back(slices.at(idx));
  }

  return poplar::concat(toConcat, dim);
}

ResizeGradOpx::ResizeGradOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<ResizeGradOp>(op);
}

void ResizeGradOpx::grow(poplar::program::Sequence &prog) const {
  auto &op   = getOp<ResizeGradOp>();
  auto input = getInTensor(ResizeGradOp::getInIndex());

  auto inShape  = op.inShape(ResizeGradOp::getInIndex());
  auto outShape = op.outShape(ResizeGradOp::getOutIndex());

  auto result = cloneNcopy(prog, input);
  for (int dimension = 0; dimension < inShape.size(); dimension++) {
    auto inDim  = inShape.at(dimension);
    auto outDim = outShape.at(dimension);
    if (inDim > outDim) {
      result = reduceDimension(
          prog, result, dimension, 1.0f / op.getFwdScales().at(dimension));
    } else if (inDim < outDim) {
      result = padDimension(prog,
                            result,
                            dimension,
                            outDim,
                            1.0f / op.getFwdScales().at(dimension));
    }
  }

  setOutTensor(ResizeGradOp::getOutIndex(), result);
}

poplar::Tensor ResizeGradOpx::reduceDimension(poplar::program::Sequence &prog,
                                              const poplar::Tensor &input,
                                              int dimension,
                                              float scale) const {
  auto slices = split(input, dimension);

  std::map<int, poplar::Tensor> resultMap;
  for (int i = 0; i < slices.size(); i++) {
    int idx = static_cast<int>(std::floor(i * scale));
    if (resultMap.find(idx) == resultMap.end()) {
      resultMap[idx] = slices[i];
    } else {
      resultMap[idx] = popops::map(graph(),
                                   popops::expr::BinaryOpType::ADD,
                                   resultMap[idx],
                                   slices[i],
                                   prog);
    }
  }

  std::vector<poplar::Tensor> toConcat;
  for (int i = 0; i < resultMap.size(); i++) {
    toConcat.push_back(resultMap.at(i));
  }
  return poplar::concat(toConcat, dimension);
}

poplar::Tensor ResizeGradOpx::padDimension(poplar::program::Sequence &prog,
                                           const poplar::Tensor &input,
                                           int dimension,
                                           int64_t newSize,
                                           float scale) const {
  auto slices = split(input, dimension);
  auto paddingTensor =
      graph().addVariable(input.elementType(), slices.at(0).shape());
  popops::zero(graph(), paddingTensor, prog, debugPrefix("zeroPadding"));

  std::vector<poplar::Tensor> toConcat(newSize, paddingTensor);
  for (int i = 0; i < slices.size(); i++) {
    int idx          = static_cast<int>(std::floor(i * scale));
    toConcat.at(idx) = slices.at(i);
  }

  return poplar::concat(toConcat, dimension);
}

namespace {
OpxCreator<ResizeOpx> resizeOpxCreator(Onnx::CustomOperators::Resize);
OpxCreator<ResizeGradOpx> resizeGradOpxCreator(Onnx::GradOperators::ResizeGrad);
} // namespace

} // namespace popx
} // namespace popart
