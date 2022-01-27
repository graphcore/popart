// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/matmul.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/matmulx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/util.hpp>

#include <snap/poplin/MatMul.hpp>
#include <popops/Reduce.hpp>

#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>

namespace popart {
namespace popx {

void MatMulOpx::appendPoplarOptionsForOp(const MatMulBaseOp &op,
                                         poplar::OptionFlags &opts) {
  auto &ir = op.getIr();

  if (op.useFullyConnectedPass()) {
    if (ir.isTraining()) {
      auto phase = op.getPhase();
      if (phase == MatMulBaseOp::Phase::Fwd) {
        opts.set("fullyConnectedPass", "TRAINING_FWD");
      } else if (phase == MatMulBaseOp::Phase::BwdLHS) {
        opts.set("fullyConnectedPass", "TRAINING_BWD");
      } else if (phase == MatMulBaseOp::Phase::BwdRHS) {
        opts.set("fullyConnectedPass", "TRAINING_WU");
      }
    } else {
      opts.set("fullyConnectedPass", "INFERENCE_FWD");
    }
  }

  if (auto prop = op.getAvailableMemoryProportion()) {
    opts.set("availableMemoryProportion", std::to_string(*prop));
  }

  {
    const auto partialsType = op.getPartialsType();
    addPartialsType(partialsType, opts);
  }
}

// Add the partials type to the poplar::OptionFlags that were computed from the
// poplar::popx::PoplarOptions.
void MatMulOpx::addPartialsType(const MatMulPartialsType &partialsType,
                                poplar::OptionFlags &opts) {
  switch (partialsType) {
  case MatMulPartialsType::HALF: {
    opts.set("partialsType", "half");
    break;
  }
  case MatMulPartialsType::FLOAT: {
    opts.set("partialsType", "float");
    break;
  }
  default: {
    throw error("Bad MatMulPartialsType {}", static_cast<int>(partialsType));
  }
  }
}

MatMulOpx::MatMulOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {
  verifyOp<MatMulOp>(op,
                     {Onnx::Operators::MatMul_1, Onnx::Operators::MatMul_9});
}

std::vector<std::size_t> MatMulOpx::onnxShapeToPoplar(const Shape &shape) {
  std::size_t m      = shape[shape.size() - 2];
  std::size_t n      = shape[shape.size() - 1];
  std::size_t stacks = std::accumulate(
      shape.begin(), shape.end() - 2, 1, std::multiplies<int64_t>());

  return {stacks, m, n};
}

std::vector<std::size_t> MatMulOpx::getOutputShape() const {
  auto matmul = getMatMulOp();
  return MatMulOpx::onnxShapeToPoplar(matmul->outInfo(0).shape());
}

static std::pair<snap::Tensor, snap::Tensor>
matInitReshape(MatMulBaseOp &matmul, snap::Tensor lhs, snap::Tensor rhs) {
  if (lhs.rank() < matmul.getExpandedLhsShape().size()) {
    lhs =
        lhs.reshape(vXtoY<int64_t, std::size_t>(matmul.getExpandedLhsShape()));
  }

  if (rhs.rank() < matmul.getExpandedRhsShape().size()) {
    rhs =
        rhs.reshape(vXtoY<int64_t, std::size_t>(matmul.getExpandedRhsShape()));
  }

  return {lhs, rhs};
}

static std::vector<std::size_t> matchRank(std::vector<std::size_t> shape,
                                          unsigned rank) {
  std::vector<std::size_t> newShape(rank, 1);

  std::copy(shape.rbegin(), shape.rend(), newShape.rbegin());

  return newShape;
}

static std::pair<snap::Tensor, snap::Tensor> matMatchRank(snap::Tensor lhs,
                                                          snap::Tensor rhs) {
  auto rank = std::max(lhs.rank(), rhs.rank());
  return {lhs.reshape(matchRank(lhs.shape(), rank)),
          rhs.reshape(matchRank(rhs.shape(), rank))};
}

static std::vector<unsigned> matDimshuffle(std::vector<std::size_t> lhsShape,
                                           std::vector<std::size_t> rhsShape) {
  std::vector<unsigned> permutation(lhsShape.size() - 2);
  boost::iota(permutation, 0);

  const auto compareDimensions = [&](unsigned dim) {
    return lhsShape[dim] == rhsShape[dim];
  };

  boost::stable_partition(permutation, compareDimensions);

  permutation.push_back(static_cast<unsigned>(lhsShape.size() - 2));
  permutation.push_back(static_cast<unsigned>(lhsShape.size() - 1));

  return permutation;
}

static std::pair<snap::Tensor, snap::Tensor> matDimshuffle(snap::Tensor lhs,
                                                           snap::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto rhsShape = rhs.shape();

  return {lhs.dimShuffle(matDimshuffle(lhsShape, rhsShape)),
          rhs.dimShuffle(matDimshuffle(lhsShape, rhsShape))};
}

static std::vector<std::size_t>
lhsReshapeGroups(std::vector<std::size_t> lhsShape,
                 std::vector<std::size_t> rhsShape) {
  auto begin = lhsShape.begin();
  auto groupEnd =
      std::mismatch(lhsShape.begin(), lhsShape.end() - 2, rhsShape.begin())
          .first;
  auto broadcastEnd = lhsShape.end() - 2;

  unsigned groupSize =
      std::accumulate(begin, groupEnd, 1, std::multiplies<std::size_t>());

  unsigned broadcastSize = std::accumulate(
      groupEnd, broadcastEnd, 1, std::multiplies<std::size_t>());

  std::vector<std::size_t> result = {groupSize, broadcastSize, 1, 1};
  std::copy(lhsShape.rbegin(), lhsShape.rbegin() + 2, result.rbegin());

  return result;
}

static std::vector<std::size_t>
rhsReshapeGroups(const std::vector<std::size_t> &lhsShape,
                 const std::vector<std::size_t> &rhsShape) {
  return lhsReshapeGroups(rhsShape, lhsShape);
}

static std::pair<snap::Tensor, snap::Tensor>
matReshapeGroups(snap::Tensor lhs, snap::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto rhsShape = rhs.shape();

  return {lhs.reshape(lhsReshapeGroups(lhsShape, rhsShape)),
          rhs.reshape(rhsReshapeGroups(lhsShape, rhsShape))};
}

static std::vector<std::size_t>
matCombineBroadcastDims(std::vector<std::size_t> shape) {
  return {shape[0], shape[1] * shape[2], shape[3]};
}

static std::pair<snap::Tensor, snap::Tensor>
matCombineBroadcastDims(snap::Tensor lhs, snap::Tensor rhs) {
  rhs = rhs.dimShuffle({0, 1, 3, 2});
  lhs = lhs.reshape(matCombineBroadcastDims(lhs.shape()));
  rhs = rhs.reshape(matCombineBroadcastDims(rhs.shape()));
  return {lhs, rhs.dimShuffle({0, 2, 1})};
}

static snap::Tensor
matSplitBroadcastDims(snap::Tensor result, snap::Tensor lhs, snap::Tensor rhs) {
  return result.reshape(
      {result.dim(0), lhs.dim(1), lhs.dim(2), rhs.dim(1), rhs.dim(3)});
}

static snap::Tensor matUnDimShuffle(snap::Tensor result) {
  return result.dimShuffle({0, 1, 3, 2, 4});
}

static snap::Tensor matExpandBroadcastDims(snap::Tensor result,
                                           snap::Tensor lhs,
                                           snap::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto rhsShape = rhs.shape();
  const auto outShape = result.shape();

  const auto itrs =
      std::mismatch(lhsShape.begin(), lhsShape.end() - 2, rhsShape.begin());

  std::vector<std::size_t> newShape;
  newShape.reserve(lhs.rank() + rhs.rank());

  std::copy(lhsShape.begin(), lhsShape.end() - 2, std::back_inserter(newShape));
  std::copy(itrs.second, rhsShape.end() - 2, std::back_inserter(newShape));
  std::copy(outShape.end() - 2, outShape.end(), std::back_inserter(newShape));

  return result.reshape(newShape);
}

static snap::Tensor
matExpandGroupDims(snap::Tensor result, snap::Tensor lhs, snap::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto rhsShape = rhs.shape();
  const auto outShape = result.shape();

  const auto offset = std::distance(
      lhsShape.begin(), boost::mismatch(lhsShape, rhs.shape()).first);

  std::vector<std::size_t> newShape;
  newShape.reserve(lhs.rank());

  std::copy(lhsShape.begin(),
            lhsShape.begin() + offset,
            std::back_inserter(newShape));
  std::copy(
      outShape.begin() + offset, outShape.end(), std::back_inserter(newShape));

  return result.reshape(newShape);
}

static snap::Tensor matInterleaveBroadcastDims(snap::Tensor result,
                                               snap::Tensor lhs,
                                               snap::Tensor rhs) {
  const auto lhsShape = lhs.shape();

  const auto offset = std::distance(
      lhsShape.begin(), boost::mismatch(lhsShape, rhs.shape()).first);

  const auto length = lhs.rank() - offset - 2;

  std::vector<unsigned> permutation(result.rank());
  boost::iota(permutation, 0);

  for (int i = 0; i < length; ++i) {
    for (int k = 0; k < 2; ++k) {
      permutation[offset + i * 2 + k] =
          static_cast<unsigned>(offset + k * length + i);
    }
  }

  return result.dimShuffle(permutation);
}

static snap::Tensor matSqueezeBroadcastDims(snap::Tensor result,
                                            snap::Tensor lhs,
                                            snap::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto offset   = std::distance(
      lhsShape.begin(), boost::mismatch(lhsShape, rhs.shape()).first);

  std::vector<std::size_t> squeezeDims;
  for (auto i = offset; i < result.rank() - 2; ++i) {
    if (result.dim(static_cast<unsigned>(i)) == 1) {
      squeezeDims.push_back(i);
    }
  }
  return result.squeeze(squeezeDims);
}

template <typename T1, typename T2>
static std::vector<T1> permute(std::vector<T1> input,
                               std::vector<T2> permutation) {
  auto output = input;

  for (int i = 0; i < output.size(); ++i) {
    output[i] = input[permutation[i]];
  }

  return output;
}

template <typename T>
static std::vector<T> invertPermutation(std::vector<T> permutation) {
  auto output = permutation;

  for (int i = 0; i < output.size(); ++i) {
    output[permutation[i]] = i;
  }

  return output;
}

static std::vector<unsigned>
matShuffleGroupDims(std::vector<std::size_t> rShape,
                    std::vector<std::size_t> lhsShape,
                    std::vector<std::size_t> rhsShape) {
  std::vector<unsigned> mapping;

  mapping.reserve(rShape.size());
  for (int i = 0; i < lhsShape.size() - 2; ++i) {
    if (lhsShape[i] == rhsShape[i]) {
      mapping.push_back(i);
    }
  }

  for (int i = 0; i < rShape.size(); ++i) {
    if (mapping.end() == boost::find(mapping, i)) {
      mapping.push_back(i);
    }
  }

  return invertPermutation(mapping);
}

static snap::Tensor
matShuffleGroupDims(snap::Tensor result, snap::Tensor lhs, snap::Tensor rhs) {
  const auto permutation =
      matShuffleGroupDims(result.shape(), lhs.shape(), rhs.shape());

  return result.dimShuffle(permutation);
}

poplar::Type MatMulOpx::getOutputType(const snap::Tensor &output) const {
  auto outputType = output.elementType();
  if (auto _outputType = getOp<MatMulOp>().getOutputType()) {
    outputType = popType(*_outputType);
  }
  return outputType;
}

void MatMulOpx::verifyCacheSizeUnchanged(size_t beforeCacheSize) const {
  bool expectedCacheSize;

  auto opts = dv_p->lowering().matmulOptions;
  appendPoplarOptionsForOp(getOp<MatMulOp>(), opts);
  auto hasFlag = [](auto &opts, auto flag) {
    for (auto &x : opts) {
      if (x.first == "fullyConnectedPass") {
        return true;
      }
    }
    return false;
  };

  if (hasFlag(opts, "fullyConnectedPass") &&
      opts.at("fullyConnectedPass") != "INFERENCE_FWD") {
    expectedCacheSize = dv_p->matmulCache.size() <= beforeCacheSize + 2;
  } else {
    expectedCacheSize = beforeCacheSize == dv_p->matmulCache.size();
  }

  if (!expectedCacheSize) {
    throw internal_error(
        "Pre-planning failed for {}. Its plan was not found in the cache",
        op_p->str());
  }
}

// Expand a matmul into a poplibs grouped matmul, following numpy rules
//
// For example,
// let `a` be a tensor with shape [2, 1, 4, 5, 1, 7, 8], and `b` be a tensor
// with shape [2, 3, 1, 5, 6, 8, 9]. We would expect an output tensor with shape
// [2, 3, 4, 5, 6, 7, 9].
void MatMulOpx::grow(snap::program::Sequence &prog) const {

  auto &matmul = getOp<MatMulOp>();

  auto a = getInTensor(MatMulOp::getLhsInIndex());
  auto b = getInTensor(MatMulOp::getRhsInIndex());

  // Makes both input tensors at least rank 3
  //
  // This doesn't change the example inputs because the
  // rank is already more than 3.
  // a' := a = [2, 1, 4, 5, 1, 7, 8]
  // b' := b = [2, 3, 1, 5, 6, 8, 9]
  auto initReshapedTs = matInitReshape(matmul, a, b);

  // Match the ranks of both tensors by prefixing their shape with 1s
  //
  // This doesn't change the example inputs because the
  // inputs already have equal rank.
  // a' := a = [2, 1, 4, 5, 1, 7, 8]
  // b' := b = [2, 3, 1, 5, 6, 8, 9]
  auto matchedRankTs =
      matMatchRank(initReshapedTs.first, initReshapedTs.second);

  // Partition the group dimensions from the broadcast dimensions
  //
  // The shapes in the given example
  // let a = [2, 1, 4, 5, 1, 7, 8],
  //     b = [2, 3, 1, 5, 6, 8, 9]
  //                                G  |    B    |
  // a' := matDimshuffle(a, b) = [2, 5 | 1, 4, 1 | 7, 8]
  // b' := matDimshuffle(a, b) = [2, 5 | 3, 1, 6 | 8, 9]
  auto dimShuffledTs = matDimshuffle(matchedRankTs.first, matchedRankTs.second);

  // Reduce the group and broadcast dimensions down to a single dimension each
  //
  // The shapes in the given example
  // let a = [2, 5, 1, 4, 1, 7, 8],
  //     b = [2, 5, 3, 1, 6, 8, 9]
  //                                  G |  B |
  // a' := matReshapeGroups(a, b) = [10 |  4 | 7, 8]
  // b' := matReshapeGroups(a, b) = [10 | 18 | 8, 9]
  auto reshapedGroupsTs =
      matReshapeGroups(dimShuffledTs.first, dimShuffledTs.second);

  // Combine the broadcast dimension into the matrix row or column dimension as
  // appropriate
  //
  // The shapes in the given example
  // let a = [10,  4, 7, 8],
  //     b = [10, 18, 8, 9]
  //                                  G
  // a' := matReshapeGroups(a, b) = [10 | 28,   8]
  // b' := matReshapeGroups(a, b) = [10 |  8, 162]
  auto combinedBroadcastTs =
      matCombineBroadcastDims(reshapedGroupsTs.first, reshapedGroupsTs.second);

  // Perform the grouped matmul
  //
  // The shapes in the given example
  // let a = [10, 28,   8],
  //     b = [10,  8, 162]
  //                        G |  M   N
  // o' := matmul(a, b) = [10 | 28, 162]

  auto opts = dv_p->lowering().matmulOptions;
  appendPoplarOptionsForOp(matmul, opts);
  auto outputType = getOutputType(combinedBroadcastTs.first);

  auto cacheSize = dv_p->matmulCache.size();
  auto outTensor =
      snap::poplin::matMulGrouped(graph(),                    // graph
                                  combinedBroadcastTs.first,  // A
                                  combinedBroadcastTs.second, // B
                                  prog,                       // prog
                                  outputType,
                                  debugContext("matmulGrouped"), // debugContext
                                  opts,                          // options
                                  &dv_p->matmulCache);           // cache

  verifyCacheSizeUnchanged(cacheSize);

  // Log the report plan
  std::stringstream ss;
  snap::poplin::matMulGroupedReportPlan(ss,
                                        graph(),
                                        combinedBroadcastTs.first.elementType(),
                                        outTensor.elementType(),
                                        combinedBroadcastTs.first.shape(),
                                        combinedBroadcastTs.second.shape(),
                                        opts,
                                        &dv_p->matmulCache);
  logging::opx::debug("Grouped Matmul {} plan", op_p->str());
  logging::log(logging::Module::opx, logging::Level::Debug, ss.str());

  // Split the broadcast dimensions from the rows and columns
  //
  // The shapes in the given example
  // let a = [10,  4, 7, 8],
  //     b = [10, 18, 8, 9]
  //     o = [10, 28, 162]
  //                                          G | B1 | M | B2 | N
  // o' := matSplitBroadcastDims(o, a, b) = [10 |  4 | 7 | 18 | 9]
  outTensor = matSplitBroadcastDims(
      outTensor, reshapedGroupsTs.first, reshapedGroupsTs.second);
  // Shuffle the column broadcast dim forward
  //
  // The shapes in the given example
  //     o = [10, 4, 7, 18, 9]
  //                                    G | B1 B2 | M  N
  // o' := matUnDimShuffle(o, a, b) = [10 | 4, 18 | 7, 9]
  outTensor = matUnDimShuffle(outTensor);

  // Expand the broadcast dimensions back to their original shape
  //
  // The shapes in the given example
  // let a = [2, 5, 1, 4, 1, 7, 8],
  //     b = [2, 5, 3, 1, 6, 8, 9]
  //     o = [10, 4, 18, 7, 9]
  //                                           G |    B1   |    B2   | M  N
  // o' := matExpandBroadcastDims(o, a, b) = [10 | 1, 4, 1 | 3, 1, 6 | 7, 9]
  outTensor = matExpandBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);
  // Interleave the broadcast dimensions that should be squeezed
  //
  // The shapes in the given example
  // let a = [2, 5, 1, 4, 1, 7, 8],
  //     b = [2, 5, 3, 1, 6, 8, 9]
  //     o = [10, 1, 4, 1, 3, 1, 6, 7, 9]
  //                                               G |         B        | M  N
  // o' := matInterleaveBroadcastDims(o, a, b) = [10 | 1, 3, 4, 1, 1, 6 | 7, 9]
  outTensor = matInterleaveBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);

  // Squeeze the broadcast dimensions
  //
  // The shapes in the given example
  // let a = [2, 5, 1, 4, 1, 7, 8],
  //     b = [2, 5, 3, 1, 6, 8, 9]
  //     o = [10, 1, 3, 4, 1, 1, 6, 7, 9]
  //                                            G |    B    | M  N
  // o' := matSqueezeBroadcastDims(o, a, b) = [10 | 3, 4, 6 | 7, 9]
  outTensor = matSqueezeBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);

  // Expand the group dimensions
  //
  // The shapes in the given example
  // let a = [2, 5, 1, 4, 1, 7, 8],
  //     b = [2, 5, 3, 1, 6, 8, 9]
  //     o = [10, 3, 4, 6, 7, 9]
  //                                        G  |    B    | M  N
  // o' := matExpandGroupDims(o, a, b) = [2, 5 | 3, 4, 6 | 7, 9]
  outTensor =
      matExpandGroupDims(outTensor, dimShuffledTs.first, dimShuffledTs.second);

  // Shuffle the group dimensions back into place
  //
  // The shapes in the given example
  // let a = [2, 1, 4, 5, 1, 7, 8],
  //     b = [2, 3, 1, 5, 6, 8, 9]
  //     o = [2, 5, 3, 4, 6, 7, 9]
  //                                                     | M  N
  // o' := matShuffleGroupDims(o, a, b) = [2, 3, 4, 5, 6 | 7, 9]
  outTensor =
      matShuffleGroupDims(outTensor, matchedRankTs.first, matchedRankTs.second);

  setOutTensor(0, outTensor.reshape(matmul.outInfo(0).shape_szt()));
}

MatMulOp *MatMulOpx::getMatMulOp() const {
  return dynamic_cast<MatMulOp *>(op_p);
}

snap::Tensor
MatMulOpx::createInputTensor(InIndex index,
                             const poplar::DebugNameAndId &dnai) const {
  auto &matmul = getOp<MatMulOp>();

  std::vector<std::size_t> lhsShape =
      vXtoY<int64_t, std::size_t>(matmul.getExpandedLhsShape());
  std::vector<std::size_t> rhsShape =
      vXtoY<int64_t, std::size_t>(matmul.getExpandedRhsShape());

  lhsShape = matchRank(
      lhsShape,
      static_cast<unsigned>(std::max(lhsShape.size(), rhsShape.size())));
  rhsShape = matchRank(
      rhsShape,
      static_cast<unsigned>(std::max(lhsShape.size(), rhsShape.size())));

  const auto permutation = matDimshuffle(lhsShape, rhsShape);
  const auto lhsShapeP   = permute(lhsShape, permutation);
  const auto rhsShapeP   = permute(rhsShape, permutation);

  const auto lhsReshapeGroupsL = [rhsShapeP](std::vector<std::size_t> shape) {
    return lhsReshapeGroups(shape, rhsShapeP);
  };

  const auto rhsReshapeGroupsL = [lhsShapeP](std::vector<std::size_t> shape) {
    return rhsReshapeGroups(lhsShapeP, shape);
  };

  lhsShape = lhsReshapeGroupsL(lhsShapeP);
  rhsShape = rhsReshapeGroupsL(rhsShapeP);

  lhsShape = matCombineBroadcastDims(lhsShape);

  std::swap(rhsShape[3], rhsShape[2]);
  rhsShape = matCombineBroadcastDims(rhsShape);
  std::swap(rhsShape[2], rhsShape[1]);

  auto opts = dv_p->lowering().matmulOptions;
  appendPoplarOptionsForOp(matmul, opts);

  if (index == MatMulOp::getLhsInIndex()) {
    auto result = snap::poplin::createMatMulGroupedInputLHS(
        graph(),
        popType(getMatMulOp()->lhsIn()->info.dataType()),
        popType(getMatMulOp()->lhsIn()->info.dataType()),
        lhsShape,
        rhsShape,
        dnai,
        opts,
        &dv_p->matmulCache);

    result = result.reshape(lhsShapeP);
    result = result.dimShuffle(invertPermutation(permutation));

    return result.reshape(matmul.lhsIn()->info.shape_szt());
  } else if (index == MatMulOp::getRhsInIndex()) {
    auto result = snap::poplin::createMatMulGroupedInputRHS(
        graph(),
        popType(getMatMulOp()->lhsIn()->info.dataType()),
        popType(getMatMulOp()->lhsIn()->info.dataType()),
        lhsShape,
        rhsShape,
        dnai,
        opts,
        &dv_p->matmulCache);

    result = result.reshape(rhsShapeP);
    result = result.dimShuffle(invertPermutation(permutation));

    return result.reshape(matmul.rhsIn()->info.shape_szt());
  } else {
    throw error("MatMulOpx::createInput invalid input index {}", index);
  }
}

InputCreatorType MatMulOpx::getInputCreatorType(InIndex) const {
  const MatMulOp *op = dynamic_cast<const MatMulOp *>(op_p);
  if (op->getCanCreateInputs()) {
    return InputCreatorType::CanCreate;
  } else {
    return InputCreatorType::Deadend;
  }
}

std::set<TensorId> MatMulOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

std::pair<snap::Tensor, snap::Tensor>
MatMulOpx::groupedMatMulInputsFromOpxInputs(MatMulBaseOp &matmul,
                                            snap::Tensor lhs,
                                            snap::Tensor rhs) {
  auto initReshapedTs = matInitReshape(matmul, lhs, rhs);

  auto matchedRankTs =
      matMatchRank(initReshapedTs.first, initReshapedTs.second);

  auto dimShuffledTs = matDimshuffle(matchedRankTs.first, matchedRankTs.second);

  auto reshapedGroupsTs =
      matReshapeGroups(dimShuffledTs.first, dimShuffledTs.second);

  auto combinedBroadcastTs =
      matCombineBroadcastDims(reshapedGroupsTs.first, reshapedGroupsTs.second);

  return combinedBroadcastTs;
}

namespace {
OpxCreator<MatMulOpx> matmulOpxCreator({Onnx::Operators::MatMul_1,
                                        Onnx::Operators::MatMul_9});
} // namespace

} // namespace popx
} // namespace popart
