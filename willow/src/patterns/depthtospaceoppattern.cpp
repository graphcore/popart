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

std::vector<int64_t> increaseRankDTS(const Shape &shapeIn,
                                     DepthToSpaceMode mode,
                                     int64_t blocksize) {

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

std::vector<int64_t> decreaseRankDTS(const Shape &shapeIn, int64_t blocksize) {
  return {shapeIn[0],
          shapeIn[1] / (blocksize * blocksize),
          shapeIn[2] * blocksize,
          shapeIn[3] * blocksize};
}

std::vector<int64_t> permDTS(DepthToSpaceMode mode) {

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

std::vector<int64_t> permSTD() {

  // Get permutation vector to transpose to:
  // [N, blocksize, blocksize, C, H/blocksize, W/blocksize]
  return {0, 3, 5, 1, 2, 4};
}

std::vector<int64_t> increaseRankSTD(const Shape &shapeIn, int64_t blocksize) {

  // We have N, C, H, W = x.shape
  // to [N, C,  H/blocksize, blocksize, W/blocksize, blocksize]
  return {shapeIn[0],
          shapeIn[1],
          shapeIn[2] / blocksize,
          blocksize,
          shapeIn[3] / blocksize,
          blocksize};
}

std::vector<int64_t> decreaseRankSTD(const Shape &shapeIn, int64_t blocksize) {
  return {shapeIn[0],
          shapeIn[1] * blocksize * blocksize,
          shapeIn[2] / blocksize,
          shapeIn[3] / blocksize};
}

} // namespace

Op *DepthSpaceOpPattern::transposeDepthSpace(const std::vector<int64_t> &perm,
                                             Tensor *input,
                                             Op *depthSpace,
                                             Op *reshape6D,
                                             Graph &graph) const {
  std::unique_ptr<TransposeOp> transposeA = std::make_unique<TransposeOp>(
      Onnx::Operators::Transpose_1,
      perm,
      Op::Settings(graph, depthSpace->name() + "_" + "Transpose"));

  Op *transpose1 = transposeA.get();
  transferBaseProperties(depthSpace, transpose1);
  graph.moveIntoGraph(std::move(transposeA));

  transpose1->connectInTensor(
      0, reshape6D->outTensor(ReshapeOp::getOutIndex())->id);
  transpose1->createAndConnectOutTensor(
      0, input->getIr().createIntermediateTensorId("transpose1"));
  transpose1->setup();

  return transpose1;
}

void DepthSpaceOpPattern::transform(const Shape &shapeIn,
                                    const std::vector<int64_t> &shape6D,
                                    int64_t blocksize,
                                    Tensor *input,
                                    Tensor *output,
                                    Op *depthSpace,
                                    Graph &graph) const {

  // reshape increase rank
  std::unique_ptr<ReshapeOp> reshape6DA = std::make_unique<ReshapeOp>(
      Onnx::Operators::Reshape_5,
      shape6D,
      Op::Settings(graph, depthSpace->name() + "_" + "Reshape"));

  Op *reshape6D = reshape6DA.get();
  transferBaseProperties(depthSpace, reshape6D);
  graph.moveIntoGraph(std::move(reshape6DA));

  reshape6D->connectInTensor(0, input->id);
  reshape6D->createAndConnectOutTensor(
      0, input->getIr().createIntermediateTensorId("reshape6D"));
  reshape6D->setup();

  // transpose
  std::vector<int64_t> perm;
  std::vector<int64_t> shape4D;
  if (depthSpace->isConvertibleTo<DepthToSpaceOp>()) {
    auto depthToSpace = static_cast<DepthToSpaceOp *>(depthSpace);
    perm              = permDTS(depthToSpace->getMode());
    shape4D           = decreaseRankDTS(shapeIn, blocksize);
  } else {
    perm    = permSTD();
    shape4D = decreaseRankSTD(shapeIn, blocksize);
  }

  Op *transpose1 =
      transposeDepthSpace(perm, input, depthSpace, reshape6D, graph);

  // reshape decrease rank
  std::unique_ptr<ReshapeOp> reshape4DB = std::make_unique<ReshapeOp>(
      Onnx::Operators::Reshape_5,
      shape4D,
      Op::Settings(graph, depthSpace->name() + "_" + "Reshape"));

  Op *reshape4D = reshape4DB.get();
  transferBaseProperties(depthSpace, reshape4D);
  graph.moveIntoGraph(std::move(reshape4DB));

  reshape4D->connectInTensor(
      0, transpose1->outTensor(TransposeOp::getOutIndex())->id);
  reshape4D->connectOutTensor(ReshapeOp::getOutIndex(), output->id);
  reshape4D->setup();
}

bool DepthToSpaceOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<DepthToSpaceOp>();
}

std::vector<const Tensor *> DepthToSpaceOpPattern::touches(Op *) const {
  return {};
}

// We have b, c, h, w = input.shape
// Reshape:
// [b, blocksize, blocksize, c / (blocksize^2), h, w] for DCR
// [b, c / (blocksize ** 2), blocksize, blocksize, h, w]) for CRD
// Transpose to:
// [b, c / (blocksize^2), h, blocksize, w, blocksize]
// Reshape:
// [b, c / blocksize^2, h * blocksize, w * blocksize]
bool DepthToSpaceOpPattern::apply(Op *op) const {
  auto &graph       = op->getGraph();
  auto depthToSpace = static_cast<DepthToSpaceOp *>(op);

  auto input  = op->inTensor(DepthToSpaceOp::getInIndex());
  auto output = op->outTensor(DepthToSpaceOp::getOutIndex());

  Shape shapeIn                = input->info.shape();
  DepthToSpaceMode mode        = depthToSpace->getMode();
  int64_t blocksize            = depthToSpace->getBlocksize();
  std::vector<int64_t> shape6D = increaseRankDTS(shapeIn, mode, blocksize);

  op->disconnectAllInputs();
  op->disconnectAllOutputs();

  transform(shapeIn, shape6D, blocksize, input, output, depthToSpace, graph);

  // Remove the DepthToSpaceOp
  op->getGraph().eraseOp(op->id);

  return true;
}

bool SpaceToDepthOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<SpaceToDepthOp>();
}

std::vector<const Tensor *> SpaceToDepthOpPattern::touches(Op *) const {
  return {};
}

// We have N, C, H, W = input.shape
// Reshape:
// [N, C,  H/blocksize, blocksize, W/blocksize, blocksize]
// Transpose to:
// [N, blocksize, blocksize, C, H/blocksize, W/blocksize]
// Reshape:
// [N, C*blocksize^2, H/blocksize, W/blocksize]
bool SpaceToDepthOpPattern::apply(Op *op) const {
  auto &graph       = op->getGraph();
  auto spaceToDepth = static_cast<SpaceToDepthOp *>(op);

  auto input  = op->inTensor(SpaceToDepthOp::getInIndex());
  auto output = op->outTensor(SpaceToDepthOp::getOutIndex());

  Shape shapeIn = input->info.shape();

  int64_t blocksize            = spaceToDepth->getBlocksize();
  std::vector<int64_t> shape6D = increaseRankSTD(shapeIn, blocksize);

  op->disconnectAllInputs();
  op->disconnectAllOutputs();

  transform(shapeIn, shape6D, blocksize, input, output, spaceToDepth, graph);

  // Remove the SpaceToDepthOp
  op->getGraph().eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<popart::DepthToSpaceOpPattern>
    DepthToSpaceOpPattern(PreAliasPatternType::DepthToSpaceOpPattern,
                          "DepthToSpaceOpPattern");

static PatternCreator<popart::SpaceToDepthOpPattern>
    SpaceToDepthOpPattern(PreAliasPatternType::SpaceToDepthOpPattern,
                          "SpaceToDepthOpPattern");

} // namespace

} // namespace popart
