// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <cmath>

#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/depthtospace.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/transpose.hpp>
#include <popart/patterns/depthtospaceoppattern.hpp>
#include <popart/patterns/pattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {
namespace {

std::vector<int64_t>
sixDimensions(const Shape &shapeIn, DepthToSpaceMode mode, int64_t blocksize) {

  // We have b, c, h, w = x.shape
  switch (mode) {
  case DepthToSpaceMode::DCR:
    // [b, blocksize, blocksize, c / (blocksize^2), h, w]
    return {shapeIn[0],
            blocksize,
            blocksize,
            shapeIn[1] / (blocksize * blocksize),
            shapeIn[2],
            shapeIn[3]};
  case DepthToSpaceMode::CRD:
    // [b, c / (blocksize ** 2), blocksize, blocksize, h, w])
    return {shapeIn[0],
            shapeIn[1] / (blocksize * blocksize),
            blocksize,
            blocksize,
            shapeIn[2],
            shapeIn[3]};
  default:
    throw error("Bad DepthToSpaceMode.");
  }
}

std::vector<int64_t> fourDimensions(const Shape &shapeIn, int64_t blocksize) {
  return {shapeIn[0],
          shapeIn[1] / (blocksize * blocksize),
          shapeIn[2] * blocksize,
          shapeIn[3] * blocksize};
}

std::vector<int64_t> transposePermutation(DepthToSpaceMode mode) {

  // Get permutation vector to transpose to:
  // [b, c / (blocksize^2), h, blocksize, w, blocksize]
  switch (mode) {
  case DepthToSpaceMode::DCR:
    return {0, 3, 4, 1, 5, 2};
  case DepthToSpaceMode::CRD:
    return {0, 1, 4, 2, 5, 3};
  default:
    throw error("Bad DepthToSpaceMode.");
  }
}

} // namespace

bool DepthToSpaceOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<DepthToSpaceOp>();
}

std::vector<const Tensor *> DepthToSpaceOpPattern::touches(Op *) const {
  return {};
}

bool DepthToSpaceOpPattern::apply(Op *op) const {
  auto &graph       = op->getGraph();
  auto depthToSpace = static_cast<DepthToSpaceOp *>(op);

  auto input  = op->inTensor(DepthToSpaceOp::getInIndex());
  auto output = op->outTensor(DepthToSpaceOp::getOutIndex());

  Shape shapeIn = input->info.shape();

  DepthToSpaceMode mode        = depthToSpace->getMode();
  int64_t blocksize            = depthToSpace->getBlocksize();
  std::vector<int64_t> shape6D = sixDimensions(shapeIn, mode, blocksize);

  op->disconnectAllInputs();
  op->disconnectAllOutputs();

  // reshape
  std::unique_ptr<ReshapeOp> reshape6DA = std::make_unique<ReshapeOp>(
      Onnx::Operators::Reshape_5,
      shape6D,
      Op::Settings(graph, depthToSpace->name() + "_" + "Reshape"));

  Op *reshape6D = reshape6DA.get();
  transferBaseProperties(depthToSpace, reshape6D);
  graph.moveIntoGraph(std::move(reshape6DA));

  reshape6D->connectInTensor(0, input->id);
  reshape6D->createAndConnectOutTensor(
      0, input->getIr().createIntermediateTensorId("reshape6D"));
  reshape6D->setup();

  // transpose to [b, c / (blocksize^2), h, blocksize, w, blocksize]
  std::vector<int64_t> perm = transposePermutation(mode);

  std::unique_ptr<TransposeOp> transposeA = std::make_unique<TransposeOp>(
      Onnx::Operators::Transpose_1,
      perm,
      Op::Settings(graph, depthToSpace->name() + "_" + "Transpose"));

  Op *transpose1 = transposeA.get();
  transferBaseProperties(depthToSpace, transpose1);
  graph.moveIntoGraph(std::move(transposeA));

  transpose1->connectInTensor(
      0, reshape6D->outTensor(ReshapeOp::getOutIndex())->id);
  transpose1->createAndConnectOutTensor(
      0, input->getIr().createIntermediateTensorId("transpose1"));
  transpose1->setup();

  std::vector<int64_t> shape4D = fourDimensions(shapeIn, blocksize);

  // reshape
  std::unique_ptr<ReshapeOp> reshape4DB = std::make_unique<ReshapeOp>(
      Onnx::Operators::Reshape_5,
      shape4D,
      Op::Settings(graph, depthToSpace->name() + "_" + "Reshape"));

  Op *reshape4D = reshape4DB.get();
  transferBaseProperties(depthToSpace, reshape4D);
  graph.moveIntoGraph(std::move(reshape4DB));

  reshape4D->connectInTensor(
      0, transpose1->outTensor(TransposeOp::getOutIndex())->id);
  reshape4D->connectOutTensor(ReshapeOp::getOutIndex(), output->id);
  reshape4D->setup();

  // Remove the DepthToSpaceOp
  op->getGraph().eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<popart::DepthToSpaceOpPattern>
    DepthToSpaceOpPattern(PreAliasPatternType::DepthToSpaceOpPattern,
                          "DepthToSpaceOpPattern");

} // namespace

} // namespace popart
