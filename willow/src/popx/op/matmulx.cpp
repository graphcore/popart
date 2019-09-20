#include <memory>
#include <popart/error.hpp>
#include <popart/op/matmul.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/matmulx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/util.hpp>

#include <poplin/MatMul.hpp>
#include <popops/Reduce.hpp>

#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>

namespace popart {
namespace popx {

PoplarOptions getPoplarOptionsForMatMul(Devicex *device,
                                        MatMulBaseOp::Phase phase) {
  PoplarOptions options;
  if (phase == MatMulBaseOp::Phase::Fwd) {
    options = device->fwdMmOptions;
  } else if (phase == MatMulBaseOp::Phase::BwdLhs) {
    options = device->bwdMmLhsOptions;
  } else if (phase == MatMulBaseOp::Phase::BwdRhs) {
    options = device->bwdMmRhsOptions;
  }
  return options;
}

MatMulOpx::MatMulOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
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

static void setMatMulOptions(MatMulBaseOp &op, poplar::OptionFlags &opts) {
  if (auto prop = op.getAvailableMemoryProportion()) {
    opts.set("availableMemoryProportion", std::to_string(*prop));
  }
}

static std::pair<poplar::Tensor, poplar::Tensor>
matInitReshape(MatMulBaseOp &matmul, poplar::Tensor lhs, poplar::Tensor rhs) {

  auto a = lhs;
  auto b = rhs;

  if (a.rank() < matmul.getExpandedLhsShape().size()) {
    a = a.reshape(vXtoY<int64_t, std::size_t>(matmul.getExpandedLhsShape()));
  }

  if (b.rank() < matmul.getExpandedRhsShape().size()) {
    b = b.reshape(vXtoY<int64_t, std::size_t>(matmul.getExpandedRhsShape()));
  }

  return {a, b};
}

static std::vector<std::size_t> matchRank(std::vector<std::size_t> shape,
                                          unsigned rank) {
  std::vector<std::size_t> newShape(rank, 1);

  std::copy(shape.rbegin(), shape.rend(), newShape.rbegin());

  return newShape;
}

static std::pair<poplar::Tensor, poplar::Tensor>
matMatchRank(poplar::Tensor lhs, poplar::Tensor rhs) {
  return {
      lhs.reshape(matchRank(lhs.shape(), std::max(lhs.rank(), rhs.rank()))),
      rhs.reshape(matchRank(rhs.shape(), std::max(lhs.rank(), rhs.rank())))};
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

static std::pair<poplar::Tensor, poplar::Tensor>
matDimshuffle(poplar::Tensor lhs, poplar::Tensor rhs) {
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
rhsReshapeGroups(std::vector<std::size_t> lhsShape,
                 std::vector<std::size_t> rhsShape) {
  return lhsReshapeGroups(rhsShape, lhsShape);
}

static std::pair<poplar::Tensor, poplar::Tensor>
matReshapeGroups(poplar::Tensor lhs, poplar::Tensor rhs) {
  const auto lhsShape = lhs.shape();
  const auto rhsShape = rhs.shape();

  return {lhs.reshape(lhsReshapeGroups(lhsShape, rhsShape)),
          rhs.reshape(rhsReshapeGroups(lhsShape, rhsShape))};
}

static std::vector<std::size_t>
matCombineBroadcastDims(std::vector<std::size_t> shape) {
  return {shape[0], shape[1] * shape[2], shape[3]};
}

static std::pair<poplar::Tensor, poplar::Tensor>
matCombineBroadcastDims(poplar::Tensor lhs, poplar::Tensor rhs) {
  rhs = rhs.dimShuffle({0, 1, 3, 2});

  lhs = lhs.reshape(matCombineBroadcastDims(lhs.shape()));
  rhs = rhs.reshape(matCombineBroadcastDims(rhs.shape()));

  return {lhs, rhs.dimShuffle({0, 2, 1})};
}

static poplar::Tensor matSplitBroadcastDims(poplar::Tensor result,
                                            poplar::Tensor lhs,
                                            poplar::Tensor rhs) {
  return result.reshape(
      {result.dim(0), lhs.dim(1), lhs.dim(2), rhs.dim(1), rhs.dim(3)});
}

static poplar::Tensor matUnDimShuffle(poplar::Tensor result) {
  return result.dimShuffle({0, 1, 3, 2, 4});
}

static poplar::Tensor matExpandBroadcastDims(poplar::Tensor result,
                                             poplar::Tensor lhs,
                                             poplar::Tensor rhs) {
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

static poplar::Tensor matExpandGroupDims(poplar::Tensor result,
                                         poplar::Tensor lhs,
                                         poplar::Tensor rhs) {
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

static poplar::Tensor matInterleaveBroadcastDims(poplar::Tensor result,
                                                 poplar::Tensor lhs,
                                                 poplar::Tensor rhs) {
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

static poplar::Tensor matSqueezeBroadcastDims(poplar::Tensor result,
                                              poplar::Tensor lhs,
                                              poplar::Tensor rhs) {
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

static poplar::Tensor matShuffleGroupDims(poplar::Tensor result,
                                          poplar::Tensor lhs,
                                          poplar::Tensor rhs) {
  const auto permutation =
      matShuffleGroupDims(result.shape(), lhs.shape(), rhs.shape());

  return result.dimShuffle(permutation);
}

static poplar::Tensor matTranspose(poplar::Tensor t) {
  std::vector<unsigned> permutation(t.rank());
  boost::iota(permutation, 0);

  std::swap(permutation[t.rank() - 2], permutation[t.rank() - 1]);

  return t.dimShuffle(permutation);
}

// Expand a matmul into a poplibs grouped matmul, following numpy rules
//
// For example,
// let `a` be a tensor with shape [2, 1, 4, 5, 1, 7, 8], and `b` be a tensor
// with shape [2, 3, 1, 5, 6, 8, 9]. We would expect an output tensor with shape
// [2, 3, 4, 5, 6, 7, 9].
void MatMulOpx::grow(poplar::program::Sequence &prog) const {

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

  auto opts =
      getPoplarOptionsForMatMul(dv_p, matmul.getPhase()).toOptionFlags();
  setMatMulOptions(matmul, opts);

  auto outTensor =
      poplin::matMulGrouped(graph(),                    // graph
                            combinedBroadcastTs.first,  // A
                            combinedBroadcastTs.second, // B
                            prog,                       // prog
                            combinedBroadcastTs.first.elementType(),
                            debugPrefix("matmulGrouped"), // debugPrefix
                            opts,                         // options
                            &dv_p->matmulCache);          // cache

  // Log the report plan
  std::stringstream ss;
  poplin::matMulGroupedReportPlan(ss,
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

poplar::Tensor MatMulOpx::createInput(InIndex index,
                                      const std::string &name) const {
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

  auto opts =
      getPoplarOptionsForMatMul(dv_p, matmul.getPhase()).toOptionFlags();
  setMatMulOptions(matmul, opts);

  if (index == MatMulOp::getLhsInIndex()) {
    auto result = poplin::createMatMulGroupedInputLHS(
        graph(),
        popType(getMatMulOp()->lhsIn()->info.dataType()),
        popType(getMatMulOp()->lhsIn()->info.dataType()),
        lhsShape,
        rhsShape,
        name,
        opts,
        &dv_p->matmulCache);

    result = result.reshape(lhsShapeP);
    result = result.dimShuffle(invertPermutation(permutation));

    return result.reshape(matmul.lhsIn()->info.shape_szt());
  } else if (index == MatMulOp::getRhsInIndex()) {
    auto result = poplin::createMatMulGroupedInputRHS(
        graph(),
        popType(getMatMulOp()->lhsIn()->info.dataType()),
        popType(getMatMulOp()->lhsIn()->info.dataType()),
        lhsShape,
        rhsShape,
        name,
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
    return InputCreatorType::CANCREATE;
  } else {
    return InputCreatorType::DEADEND;
  }
}

bool MatMulOpx::createsEquiv(int ind0, const Opx *opx1, int ind1) const {
  if (opx1->op_p->opid != Onnx::Operators::MatMul_1 &&
      opx1->op_p->opid != Onnx::Operators::MatMul_9)
    return false;

  if (ind0 != ind1)
    return false;

  // Test that the shapes and types of inputs and outputs of the two ops are the
  // same
  // TODO : Could optimize this to not use the either lhs or rhs
  const MatMulOpx *rhs = dynamic_cast<const MatMulOpx *>(opx1);
  if (getMatMulOp()->lhsIn()->info != rhs->getMatMulOp()->lhsIn()->info ||
      getMatMulOp()->rhsIn()->info != rhs->getMatMulOp()->rhsIn()->info ||
      getMatMulOp()->out()->info != rhs->getMatMulOp()->out()->info) {
    return false;
  }

  return true;
}

std::vector<TensorId> MatMulOpx::mustExistBeforeCreate(InIndex) const {
  return {};
}

MatMulLhsGradOpx::MatMulLhsGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<MatMulLhsGradOp>(op, Onnx::GradOperators::MatMulLhsGrad);
}

static poplar::Tensor reduceResult(poplar::Graph &g,
                                   poplar::Tensor t,
                                   std::vector<std::size_t> shape,
                                   poplar::program::Sequence &prog) {
  std::vector<std::size_t> axes;

  auto t_shape = t.shape();
  auto r_shape = shape;

  if (shape != t.shape()) {

    // First remove any leading '1' dimension with a squeeze
    std::vector<std::size_t> squeezeDims;
    for (auto i = 0; i < t.shape().size() - r_shape.size(); ++i) {
      if (t.shape()[i] == 1) {
        squeezeDims.push_back(i);
      } else {
        break;
      }
    }

    if (squeezeDims.size() > 0) {
      logging::opx::debug(
          "Reducing {} {} to {}. (dims {})", t, t_shape, r_shape, squeezeDims);
      t = t.squeeze(squeezeDims);
    }
  }

  // If the shapes are still not the same then use the reduce
  if (shape != t.shape()) {

    if (shape.size() == 1) {

      // First remove any ending '1' dimension with a squeeze
      std::vector<std::size_t> squeezeDims;
      for (auto i = t.shape().size() - 1; i > (r_shape.size() - 1); --i) {
        if (t.shape()[i] == 1) {
          squeezeDims.push_back(i);
        } else {
          break;
        }
      }

      if (squeezeDims.size() > 0) {
        logging::opx::debug("Reducing {} {} to {}. (dims {})",
                            t,
                            t_shape,
                            r_shape,
                            squeezeDims);
        t = t.squeeze(squeezeDims);
      }
    }

    // If the shapes are still not the same then use the reduce
    if (shape != t.shape()) {

      t_shape = t.shape();

      while (t_shape.size() < r_shape.size()) {
        t_shape.insert(t_shape.begin(), 1);
      }

      t = t.reshape(t_shape);

      while (t_shape.size() > r_shape.size()) {
        r_shape.insert(r_shape.begin(), 1);
      }

      axes.reserve(t.rank());
      for (int i = 0; i < t.rank(); ++i) {
        if (t.dim(i) != r_shape[i]) {
          axes.push_back(i);
        }
      }

      return popops::reduce(g, t, axes, {popops::Operation::ADD}, prog)
          .reshape(shape);
    } else {
      return t;
    }
  } else {
    return t;
  }
}

void MatMulLhsGradOpx::grow(poplar::program::Sequence &prog) const {

  auto &matMulLhsGrad = getOp<MatMulLhsGradOp>();

  auto a = getInTensor(MatMulLhsGradOp::getGradInIndex());
  auto b = getInTensor(MatMulLhsGradOp::getRhsInIndex());

  auto initReshapedTs = matInitReshape(matMulLhsGrad, a, b);

  auto matchedRankTs =
      matMatchRank(initReshapedTs.first, initReshapedTs.second);

  matchedRankTs.second = matTranspose(matchedRankTs.second);

  auto dimShuffledTs = matDimshuffle(matchedRankTs.first, matchedRankTs.second);

  auto reshapedGroupsTs =
      matReshapeGroups(dimShuffledTs.first, dimShuffledTs.second);

  auto combinedBroadcastTs =
      matCombineBroadcastDims(reshapedGroupsTs.first, reshapedGroupsTs.second);

  auto opts =
      getPoplarOptionsForMatMul(dv_p, matMulLhsGrad.getPhase()).toOptionFlags();
  setMatMulOptions(matMulLhsGrad, opts);

  auto outTensor =
      poplin::matMulGrouped(graph(),                    // graph
                            combinedBroadcastTs.first,  // A
                            combinedBroadcastTs.second, // B
                            prog,                       // prog
                            combinedBroadcastTs.first.elementType(),
                            debugPrefix("matmulGrouped"), // debugPrefix
                            opts,                         // options
                            &dv_p->matmulCache);          // cache

  // Log the report plan
  std::stringstream ss;
  poplin::matMulGroupedReportPlan(ss,
                                  graph(),
                                  combinedBroadcastTs.first.elementType(),
                                  outTensor.elementType(),
                                  combinedBroadcastTs.first.shape(),
                                  combinedBroadcastTs.second.shape(),
                                  opts,
                                  &dv_p->matmulCache);
  logging::opx::debug("Grouped Matmul {} plan", op_p->str());
  logging::log(logging::Module::opx, logging::Level::Debug, ss.str());

  outTensor = matSplitBroadcastDims(
      outTensor, reshapedGroupsTs.first, reshapedGroupsTs.second);

  outTensor = matUnDimShuffle(outTensor);

  outTensor = matExpandBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);

  outTensor = matInterleaveBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);

  outTensor = matSqueezeBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);

  outTensor =
      matExpandGroupDims(outTensor, dimShuffledTs.first, dimShuffledTs.second);

  outTensor =
      matShuffleGroupDims(outTensor, matchedRankTs.first, matchedRankTs.second);

  setOutTensor(0,
               reduceResult(graph(), outTensor, outInfo(0).shape_szt(), prog));
}

MatMulLhsGradOp *MatMulLhsGradOpx::getMatMulLhsGradOp() const {
  return dynamic_cast<MatMulLhsGradOp *>(op_p);
}

std::vector<std::size_t> MatMulLhsGradOpx::getGradInputShape() const {
  return MatMulOpx::onnxShapeToPoplar(getGradInputBroadcastShape());
}

Shape MatMulLhsGradOpx::getGradInputBroadcastShape() const {
  auto matmul_lgrad = getMatMulLhsGradOp();
  return matmul_lgrad->getGradInputShape();
}

std::vector<std::size_t> MatMulLhsGradOpx::getRhsInputShape() const {
  return MatMulOpx::onnxShapeToPoplar(getRhsInputBroadcastShape());
}

Shape MatMulLhsGradOpx::getRhsInputBroadcastShape() const {
  auto matmul_lgrad = getMatMulLhsGradOp();

  const auto rhs_shape = matmul_lgrad->getRhsInputShape();
  auto grad_shape      = matmul_lgrad->getGradInputShape();

  std::copy(rhs_shape.end() - 2, rhs_shape.end(), grad_shape.end() - 2);

  return grad_shape;
}

std::vector<std::size_t> MatMulLhsGradOpx::getOutputShape() const {
  auto matmul_lgrad = getMatMulLhsGradOp();

  const auto rhs_shape = matmul_lgrad->getRhsInputShape();
  auto grad_shape      = matmul_lgrad->getGradInputShape();

  grad_shape.back() = rhs_shape[rhs_shape.size() - 2];

  return vXtoY<int64_t, std::size_t>(grad_shape);
}

std::vector<std::size_t> MatMulLhsGradOpx::getOutputReductionAxes() const {
  auto matmul_lgrad = getMatMulLhsGradOp();

  return vXtoY<int64_t, std::size_t>(
      npReductionAxis(matmul_lgrad->getOutputShape(),
                      vXtoY<std::size_t, int64_t>(getOutputShape())));
}

MatMulRhsGradOpx::MatMulRhsGradOpx(Op *op, Devicex *devicex)
    : Opx(op, devicex) {
  verifyOp<MatMulRhsGradOp>(op, Onnx::GradOperators::MatMulRhsGrad);
}

MatMulRhsGradOp *MatMulRhsGradOpx::getMatMulRhsGradOp() const {
  return dynamic_cast<MatMulRhsGradOp *>(op_p);
}

void MatMulRhsGradOpx::grow(poplar::program::Sequence &prog) const {

  auto &matMulRhsGrad = getOp<MatMulRhsGradOp>();

  auto a = getInTensor(MatMulRhsGradOp::getLhsInIndex());
  auto b = getInTensor(MatMulRhsGradOp::getGradInIndex());

  auto initReshapedTs = matInitReshape(matMulRhsGrad, a, b);

  auto matchedRankTs =
      matMatchRank(initReshapedTs.first, initReshapedTs.second);

  matchedRankTs.first = matTranspose(matchedRankTs.first);

  auto dimShuffledTs = matDimshuffle(matchedRankTs.first, matchedRankTs.second);

  auto reshapedGroupsTs =
      matReshapeGroups(dimShuffledTs.first, dimShuffledTs.second);

  auto combinedBroadcastTs =
      matCombineBroadcastDims(reshapedGroupsTs.first, reshapedGroupsTs.second);

  auto opts =
      getPoplarOptionsForMatMul(dv_p, matMulRhsGrad.getPhase()).toOptionFlags();
  setMatMulOptions(matMulRhsGrad, opts);

  auto outTensor =
      poplin::matMulGrouped(graph(),                    // graph
                            combinedBroadcastTs.first,  // A
                            combinedBroadcastTs.second, // B
                            prog,                       // prog
                            combinedBroadcastTs.first.elementType(),
                            debugPrefix("matmulGrouped"), // debugPrefix
                            opts,                         // options
                            &dv_p->matmulCache);          // cache

  // Log the report plan
  std::stringstream ss;
  poplin::matMulGroupedReportPlan(ss,
                                  graph(),
                                  combinedBroadcastTs.first.elementType(),
                                  outTensor.elementType(),
                                  combinedBroadcastTs.first.shape(),
                                  combinedBroadcastTs.second.shape(),
                                  opts,
                                  &dv_p->matmulCache);
  logging::opx::debug("Grouped Matmul {} plan", op_p->str());
  logging::log(logging::Module::opx, logging::Level::Debug, ss.str());

  outTensor = matSplitBroadcastDims(
      outTensor, reshapedGroupsTs.first, reshapedGroupsTs.second);

  outTensor = matUnDimShuffle(outTensor);

  outTensor = matExpandBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);

  outTensor = matInterleaveBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);

  outTensor = matSqueezeBroadcastDims(
      outTensor, dimShuffledTs.first, dimShuffledTs.second);

  outTensor =
      matExpandGroupDims(outTensor, dimShuffledTs.first, dimShuffledTs.second);

  outTensor =
      matShuffleGroupDims(outTensor, matchedRankTs.first, matchedRankTs.second);

  setOutTensor(0,
               reduceResult(graph(), outTensor, outInfo(0).shape_szt(), prog));
}

std::vector<std::size_t> MatMulRhsGradOpx::getLhsInputShape() const {
  return MatMulOpx::onnxShapeToPoplar(getLhsInputBroadcastShape());
}

Shape MatMulRhsGradOpx::getLhsInputBroadcastShape() const {
  auto matmul_rgrad = getMatMulRhsGradOp();

  const auto lhs_shape = matmul_rgrad->getLhsInputShape();
  auto grad_shape      = matmul_rgrad->getGradInputShape();

  std::copy(lhs_shape.end() - 2, lhs_shape.end(), grad_shape.end() - 2);

  return grad_shape;
}

std::vector<std::size_t> MatMulRhsGradOpx::getGradInputShape() const {
  return MatMulOpx::onnxShapeToPoplar(getGradInputBroadcastShape());
}

Shape MatMulRhsGradOpx::getGradInputBroadcastShape() const {
  auto matmul_rgrad = getMatMulRhsGradOp();
  return matmul_rgrad->getGradInputShape();
}

std::vector<std::size_t> MatMulRhsGradOpx::getOutputShape() const {
  auto matmul_rgrad = getMatMulRhsGradOp();

  const auto lhs_shape = matmul_rgrad->getLhsInputShape();
  auto grad_shape      = matmul_rgrad->getGradInputShape();

  grad_shape[grad_shape.size() - 2] = lhs_shape.back();

  return vXtoY<int64_t, std::size_t>(grad_shape);
}

std::vector<std::size_t> MatMulRhsGradOpx::getOutputReductionAxes() const {
  auto matmul_rgrad = getMatMulRhsGradOp();

  return vXtoY<int64_t, std::size_t>(
      npReductionAxis(matmul_rgrad->getOutputShape(),
                      vXtoY<std::size_t, int64_t>(getOutputShape())));
}

namespace {
OpxCreator<MatMulOpx> matmulOpxCreator({Onnx::Operators::MatMul_1,
                                        Onnx::Operators::MatMul_9});

OpxCreator<MatMulLhsGradOpx>
    matmulLhsGradOpxCreator(Onnx::GradOperators::MatMulLhsGrad);
OpxCreator<MatMulRhsGradOpx>
    matmulRhsGradOpxCreator(Onnx::GradOperators::MatMulRhsGrad);
} // namespace

} // namespace popx
} // namespace popart
