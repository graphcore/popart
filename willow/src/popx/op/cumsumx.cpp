// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <vector>
#include <poplin/MatMul.hpp>
#include <poputil/TileMapping.hpp>
#include <popart/op/cumsum.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/cumsumx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

CumSumOpx::CumSumOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<CumSumOp>(op, {Onnx::Operators::CumSum_11});

  axis      = dynamic_cast<CumSumOp *>(op)->getAxis();
  exclusive = dynamic_cast<CumSumOp *>(op)->getExclusive();
  reverse   = dynamic_cast<CumSumOp *>(op)->getReverse();
}

// We reshape input tensor to 2D, where one axis is the axis to accumulate on.
// Before that we must shuffle the cumulative axis to be the furthest right.
// Then we create a triangular matrix of size of T*T where T is the length of
// the accumulating dimension.
// We multiply the triangular matrix with the input matrix which effectively
// gives us cumulative sum. We shuffle and reshape back to get the final
// result. We expect good performance as MatMul is highly optimised on the IPU.
void CumSumOpx::grow(poplar::program::Sequence &prog) const {
  CheckAxisValue();
  auto x                          = getInTensor(CumSumOp::xInIndex());
  std::vector<std::size_t> xShape = x.shape();
  int64_t axisNN                  = ToNonNegativeAxis(axis);
  std::size_t xMulDim0            = x.dim(axisNN);
  std::size_t xMulDim1 =
      std::accumulate(
          xShape.begin(), xShape.end(), 1, std::multiplies<std::size_t>()) /
      xMulDim0;

  auto xShapeSize = x.shape().size();
  std::vector<unsigned> perm(xShapeSize);
  for (unsigned i = 0; i < perm.size(); i++) {
    perm[i] = i;
  }
  perm[axisNN]         = xShapeSize - 1;
  perm[xShapeSize - 1] = axisNN;

  x                                     = x.dimShuffle(perm);
  std::vector<std::size_t> xMiddleShape = x.shape();
  x                                     = x.reshape({xMulDim1, xMulDim0});

  poplar::Tensor triangularM = TriangularMatrix(xMulDim0);

  x = poplin::matMul(graph(), x, triangularM, prog, debugPrefix("cumsum_mul"));
  x = x.reshape(xMiddleShape);
  x = x.dimShuffle(perm);

  setOutTensor(CumSumOp::outIndex(), x);
}

// Four triangular types for attributes:
// exclusive/reverse:  1) F/F 2) T/F 3) F/T 4) T/T
//   triangular type:     11     01     10     00
//                        01     00     11     10
poplar::Tensor CumSumOpx::TriangularMatrix(std::size_t triangularSize) const {
  poplar::Tensor one  = graph().addConstant(poplar::FLOAT, {1}, 1.f, "one");
  poplar::Tensor zero = graph().addConstant(poplar::FLOAT, {1}, 0.f, "zero");
  graph().setTileMapping(one, 0);
  graph().setTileMapping(zero, 0);

  std::vector<poplar::Tensor> pieces;
  for (int k = 0; k < triangularSize; k++) {
    int bound = k;
    if (exclusive == 1) {
      bound += 1;
    }
    for (int i = 0; i < bound; i++) {
      pieces.push_back(zero);
    }
    for (int j = 0; j < triangularSize - bound; j++) {
      pieces.push_back(one);
    }
  }

  if (reverse == 1) {
    std::reverse(pieces.begin(), pieces.end());
  }

  poplar::Tensor triangularM = poplar::concat(pieces);
  triangularM = triangularM.reshape({triangularSize, triangularSize});

  return triangularM;
}

void CumSumOpx::CheckAxisValue() const {

  unsigned xRank1 = getInTensor(CumSumOp::xInIndex()).rank();
  int64_t xRank   = static_cast<int64_t>(xRank1);

  // Axis value must be in the range [-rank(x), rank(x)-1].
  if ((axis < -xRank) || (axis > xRank - 1)) {
    throw error("CumSumOpx op, 'axis' value out of range.");
  }
}

int64_t CumSumOpx::ToNonNegativeAxis(int64_t ax) const {
  if (ax < 0) {
    unsigned xRank1 = getInTensor(CumSumOp::xInIndex()).rank();
    int64_t xRank   = static_cast<int64_t>(xRank1);

    return xRank + ax;
  } else {
    return ax;
  }
}

namespace {
OpxCreator<CumSumOpx> cumSumOpxCreator(Onnx::Operators::CumSum_11);
} // namespace

} // namespace popx
} // namespace popart
