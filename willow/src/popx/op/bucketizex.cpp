// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

#include <popops/Cast.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/Fill.hpp>
#include <popops/Loop.hpp>

#include <popart/graphcoreoperators.hpp>
#include <popart/op/bucketize.hpp>
#include <popart/popx/op/bucketizex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

using namespace popops::expr;

namespace {

struct Range {
  poplar::Tensor lowerBound;
  poplar::Tensor upperBound;
};

Range createBoundariesRange(poplar::Graph &graph,
                            poplar::program::Sequence &prog,
                            poplar::Tensor &boundaries,
                            const poplar::Tensor &lowerBoundIdx,
                            const poplar::Tensor &upperBoundIdx) {
  static const std::vector<std::size_t> dims  = {0};
  static const std::vector<std::size_t> sizes = {1};

  return {popops::dynamicSlice(graph,
                               boundaries,
                               lowerBoundIdx,
                               dims,
                               sizes,
                               prog,
                               "/bucketizeLowerBoundSlice"),
          popops::dynamicSlice(graph,
                               boundaries,
                               upperBoundIdx,
                               dims,
                               sizes,
                               prog,
                               "/bucketizeUpperBoundSlice")};
}

auto createBucketizeLoopExpr(bool right = false) {

  if (right)
    return Select(Select(_4, _1, Gte(_5, _2)), _1, Lt(_5, _3));

  return Select(Select(_4, _1, Gt(_5, _2)), _1, Lte(_5, _3));
}

auto createBucketizeFirstStepExpr(bool right = false) {
  const auto bucketId = Const(0);

  if (right)
    return Select(bucketId, _1, Lt(_2, _3));

  return Select(bucketId, _1, Lte(_2, _3));
}

} // namespace

Bucketizex::Bucketizex(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<BucketizeOp>(op, {Onnx::CustomOperators::Bucketize});
}

bool Bucketizex::outputCreatedExternally(OutIndex index) const { return true; }

void Bucketizex::grow(poplar::program::Sequence &prog) const {
  const auto &srop   = getOp<BucketizeOp>();
  const bool isRight = srop.isRight();

  poplar::Tensor input      = getInTensor(BucketizeOp::inIndex());
  poplar::Tensor boundaries = getInTensor(BucketizeOp::boundariesInIndex());
  auto output               = getOutTensor(BucketizeOp::outIndex());
  auto &graph               = this->graph();

  static constexpr unsigned loopBegin = 0;
  const unsigned lastBucketId         = boundaries.dim(0);
  const unsigned loopEnd              = lastBucketId - 1;

  static constexpr unsigned step = 1;

  popops::fill(
      graph, output, prog, static_cast<int>(lastBucketId), "/bucketizeFill");

  const auto firstStepUpperBound = boundaries.slice(0, 1);

  popops::mapInPlace(graph,
                     createBucketizeFirstStepExpr(isRight),
                     {output, input, firstStepUpperBound},
                     prog,
                     "/bucketizeFirstElem");

  if (loopEnd > loopBegin) {
    prog.add(popops::countedLoop(
        graph,
        loopBegin,
        loopEnd,
        step,
        [&](const poplar::Tensor &idx) {
          poplar::program::Sequence loopBody;

          // Iteration in reverse order
          auto upperBoundIdx = popops::sub(
              graph, loopEnd, idx, loopBody, "/bucketizeUpperBoundIdxCalc");
          auto lowerBoundIdx = popops::sub(graph,
                                           upperBoundIdx,
                                           1u,
                                           loopBody,
                                           "/bucketizeLowerBoundIdxCalc");

          const auto boundariesRange = createBoundariesRange(
              graph, loopBody, boundaries, lowerBoundIdx, upperBoundIdx);

          const auto bucketId = popops::cast(graph,
                                             upperBoundIdx,
                                             poplar::INT,
                                             loopBody,
                                             "/bucketizeBucketId");

          popops::mapInPlace(graph,
                             createBucketizeLoopExpr(isRight),
                             {output,
                              boundariesRange.lowerBound,
                              boundariesRange.upperBound,
                              bucketId,
                              input},
                             loopBody,
                             "/bucketizeLoopBody");
          return loopBody;
        },
        "/bucketizeLoop"));
  }
}

namespace {
OpxCreator<Bucketizex> bucketizeOpxCreator(Onnx::CustomOperators::Bucketize);
} // namespace

} // namespace popx
} // namespace popart
