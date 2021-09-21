// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poplar/Tensor.hpp>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>
#include <popart/op/cumsum.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/cumsumx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {
namespace {

// Four triangular types for attributes:
// exclusive/reverse:  1) F/F 2) T/F 3) F/T 4) T/T
//   triangular type:     11     01     10     00
//                        01     00     11     10
poplar::Tensor triangularMatrix(const PopOpx &opx,
                                std::size_t triangularSize_,
                                bool exclusive_,
                                bool reverse_,
                                bool transpose_ = false) {
  const poplar::Tensor one =
      opx.getConst(poplar::FLOAT, {1}, 1.f, "one").getPoplarTensor();
  const poplar::Tensor zero =
      opx.getConst(poplar::FLOAT, {1}, 0.f, "zero").getPoplarTensor();

  std::vector<poplar::Tensor> pieces;
  for (int k = 0; k < triangularSize_; k++) {
    int bound = k;
    if (exclusive_) {
      bound += 1;
    }
    for (int i = 0; i < bound; i++) {
      pieces.push_back(zero);
    }
    for (int j = 0; j < triangularSize_ - bound; j++) {
      pieces.push_back(one);
    }
  }

  if (reverse_) {
    std::reverse(pieces.begin(), pieces.end());
  }

  poplar::Tensor triangularM = poplar::concat(pieces);
  triangularM = triangularM.reshape({triangularSize_, triangularSize_});

  if (transpose_) {
    triangularM = triangularM.transpose();
  }

  return triangularM;
}

int64_t toNonNegativeAxis(int64_t axis_, unsigned xRank_) {
  if (axis_ < 0) {
    int64_t xRank2 = static_cast<int64_t>(xRank_);

    return xRank2 + axis_;
  } else {
    return axis_;
  }
}

std::size_t secondDimension(std::size_t xMulDim0,
                            const std::vector<std::size_t> &xShape) {
  return std::accumulate(
             xShape.begin(), xShape.end(), 1, std::multiplies<std::size_t>()) /
         xMulDim0;
}

std::vector<unsigned> furthestRightPermutation(std::size_t xShapeSize,
                                               int64_t axisNN) {
  std::vector<unsigned> perm(xShapeSize);
  for (unsigned i = 0; i < perm.size(); i++) {
    perm[i] = i;
  }
  perm[axisNN]         = xShapeSize - 1;
  perm[xShapeSize - 1] = axisNN;

  return perm;
}

void checkAxisValue(int64_t axis_, unsigned xRank_) {
  int64_t xRank = static_cast<int64_t>(xRank_);

  // Axis value must be in the range [-rank(x), rank(x)-1].
  if ((axis_ < -xRank) || (axis_ > xRank - 1)) {
    throw error("CumSumOpx op, 'axis' value out of range.");
  }
}

} // namespace

CumSumOpx::CumSumOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<CumSumOp>(op, {Onnx::Operators::CumSum_11});
}

// We reshape input tensor to 2D, where one axis is the axis to accumulate on.
// Before that we must shuffle the cumulative axis to be the furthest right.
// Then we create a triangular matrix of size of T*T where T is the length of
// the accumulating dimension.
// We multiply the triangular matrix with the input matrix which effectively
// gives us cumulative sum. We shuffle and reshape back to get the final
// result. We expect good performance as MatMul is highly optimised on the IPU.
void CumSumOpx::grow(poplar::program::Sequence &prog) const {

  const auto &op       = getOp<CumSumOp>();
  const int64_t axis   = op.getAxis();
  const bool exclusive = op.getExclusive();
  const bool reverse   = op.getReverse();

  auto x = getInTensor(CumSumOp::xInIndex()).getPoplarTensor();
  checkAxisValue(axis, x.rank());
  const std::vector<std::size_t> xShape = x.shape();
  int64_t axisNN                        = toNonNegativeAxis(axis, x.rank());
  std::size_t xMulDim0                  = x.dim(axisNN);
  std::size_t xMulDim1                  = secondDimension(xMulDim0, xShape);

  const std::vector<unsigned> perm =
      furthestRightPermutation(x.shape().size(), axisNN);

  x                                           = x.dimShuffle(perm);
  const std::vector<std::size_t> xMiddleShape = x.shape();
  x                                           = x.reshape({xMulDim1, xMulDim0});

  poplar::Tensor triangularM =
      triangularMatrix(*this, xMulDim0, exclusive, reverse);

  x = poplin::matMul(graph().getPoplarGraph(),
                     x,
                     triangularM,
                     prog,
                     debugContext("cumsum_mul"));
  x = x.reshape(xMiddleShape);
  x = x.dimShuffle(perm);

  setOutTensor(CumSumOp::outIndex(), snap::Tensor{x, graph()});
}

CumSumGradOpx::CumSumGradOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<CumSumGradOp>(op, Onnx::GradOperators::CumSumGrad);
}

// The gradient of cumulative sum is a cumulative sum from the right.
// dCumulativeSum(dOut) = reverse(CumulativeSum(reverse(dOut))
// This can be interpreted as the MatMul gradient where the
// triangular matrix has been transposed.
void CumSumGradOpx::grow(poplar::program::Sequence &prog) const {

  const auto &op       = getOp<CumSumGradOp>();
  const int64_t axis   = op.getAxis();
  const bool exclusive = op.getExclusive();
  const bool reverse   = op.getReverse();

  auto dx = getInTensor(CumSumGradOp::outGradXInIndex()).getPoplarTensor();
  const std::vector<std::size_t> dxShape = dx.shape();
  int64_t axisNN                         = toNonNegativeAxis(axis, dx.rank());
  std::size_t xMulDim0                   = dx.dim(axisNN);
  std::size_t xMulDim1                   = secondDimension(xMulDim0, dxShape);

  const std::vector<unsigned> perm =
      furthestRightPermutation(dx.shape().size(), axisNN);

  dx                                          = dx.dimShuffle(perm);
  const std::vector<std::size_t> xMiddleShape = dx.shape();
  dx = dx.reshape({xMulDim1, xMulDim0});

  poplar::Tensor triangularM =
      triangularMatrix(*this, xMulDim0, exclusive, reverse, true);

  dx = poplin::matMul(graph().getPoplarGraph(),
                      dx,
                      triangularM,
                      prog,
                      debugContext("cumsum_mul"));
  dx = dx.reshape(xMiddleShape);
  dx = dx.dimShuffle(perm);

  setOutTensor(CumSumGradOp::outIndex(), snap::Tensor{dx, graph()});
}

namespace {
OpxCreator<CumSumOpx> cumSumOpxCreator(Onnx::Operators::CumSum_11);
OpxCreator<CumSumGradOpx> cumSumGradOpxCreator(Onnx::GradOperators::CumSumGrad);
} // namespace

} // namespace popx
} // namespace popart
