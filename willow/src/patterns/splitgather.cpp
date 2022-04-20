// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/integer/common_factor.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/transpose.hpp>
#include <popart/patterns/splitgather.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include "popart/basicoptionals.hpp"
#include "popart/devicemanager.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
struct OperatorIdentifier;

// Private gather op class
namespace {
class SplitGatherOp : public GatherOp {
public:
  using GatherOp::GatherOp;
};
} // namespace

bool SplitGatherPattern::matches(Op *op) const {
  // Isn't a gather op
  if (!op->isConvertibleTo<GatherOp>()) {
    return false;
  }

  // Is an already split gather op
  if (op->isConvertibleTo<SplitGatherOp>()) {
    return false;
  }

  // The op doesn't have a vgraph id
  if (!op->hasVirtualGraphId()) {
    return false;
  }

  if (op->input->tensor(GatherOp::dataInIndex())->tensorType() !=
      TensorType::Const) {
    return false;
  }

  auto gather = dynamic_cast<GatherOp *>(op);

  const auto inputShape = gather->inShape(GatherOp::dataInIndex());
  const auto axis       = gather->getAxis();

  const int64_t virtualGraphCount = op->getIr().getDeviceInfo()->getNumIpus();

  // We aren't using vgraphs
  if (!op->getIr().virtualGraphsEnabled()) {
    return false;
  }

  // We don't have any vgraphs to split on
  if (virtualGraphCount < 2) {
    return false;
  }

  const auto numElements = std::accumulate(
      inputShape.begin(), inputShape.end(), 1, std::multiplies<int64_t>());

  const auto split =
      boost::integer::gcd(virtualGraphCount, numElements / inputShape[axis]);

  // We won't split the gather
  if (split < 2) {
    return false;
  }

  return true;
}

std::vector<const Tensor *> SplitGatherPattern::touches(Op *) const {
  return {};
}

static std::unique_ptr<TransposeOp>
canonicalizeTranspose(const int64_t axis,
                      const Shape &inputShape,
                      const Op::Settings &settings) {
  std::vector<int64_t> permutation(inputShape.size());
  std::iota(permutation.begin(), permutation.end(), 0);
  std::swap(permutation.front(), permutation[axis]);

  return std::make_unique<TransposeOp>(
      Onnx::Operators::Transpose_1, permutation, settings);
}

static std::unique_ptr<ReshapeOp>
canonicalizeShape(const int64_t axis,
                  const Shape &inputShape,
                  const Op::Settings &settings) {
  const auto numElements = std::accumulate(
      inputShape.begin(), inputShape.end(), 1, std::multiplies<int64_t>());

  const Shape canonShape = {inputShape[axis], numElements / inputShape[axis]};
  return std::make_unique<ReshapeOp>(
      Onnx::Operators::Reshape_5, canonShape, settings);
}

static std::vector<std::unique_ptr<SliceOp>>
createSlices(const int64_t count, const int64_t stride, Op::Settings settings) {
  std::vector<std::unique_ptr<SliceOp>> slices;
  slices.reserve(count);

  for (int i = 0; i < count; ++i) {
    settings.vgraphId = i;

    const std::vector<int64_t> starts_ = {i * stride};
    const std::vector<int64_t> ends_   = {(i + 1) * stride};
    const std::vector<int64_t> axes_   = {1};
    const std::vector<int64_t> steps_  = {};

    slices.push_back(std::make_unique<SliceOp>(
        Onnx::Operators::Slice_1, starts_, ends_, axes_, steps_, settings));
  }

  return slices;
}

static std::vector<std::unique_ptr<SplitGatherOp>>
createGathers(const int64_t count,
              Op::Settings settings,
              const nonstd::optional<float> &availMemProp) {
  std::vector<std::unique_ptr<SplitGatherOp>> gathers;
  gathers.reserve(count);

  for (int i = 0; i < count; ++i) {
    settings.vgraphId = i;
    gathers.push_back(std::make_unique<SplitGatherOp>(
        Onnx::Operators::Gather_1, 0, settings, availMemProp));
  }

  return gathers;
}

static std::unique_ptr<ReshapeOp>
decanonicalizeShape(const int64_t axis,
                    Shape origShape,
                    Shape indicesShape,
                    const Op::Settings &settings) {
  std::swap(origShape.front(), origShape[axis]);
  origShape.erase(origShape.begin());
  origShape.insert(origShape.begin(), indicesShape.begin(), indicesShape.end());

  return std::make_unique<ReshapeOp>(
      Onnx::Operators::Reshape_5, origShape, settings);
}

static std::unique_ptr<TransposeOp>
decanonicalizeTranspose(const int64_t axis,
                        Shape inputShape,
                        Shape indicesShape,
                        const Op::Settings &settings) {
  std::vector<int64_t> permutation(inputShape.size() + indicesShape.size() - 1);
  std::iota(permutation.begin(), permutation.end(), 0);
  std::rotate(permutation.begin(),
              permutation.begin() + indicesShape.size(),
              permutation.begin() + indicesShape.size() + axis);

  return std::make_unique<TransposeOp>(
      Onnx::Operators::Transpose_1, permutation, settings);
}

bool SplitGatherPattern::apply(Op *op) const {
  auto gather = dynamic_cast<GatherOp *>(op);

  const auto inputShape           = gather->inShape(GatherOp::dataInIndex());
  const auto axis                 = gather->getAxis();
  const auto availMemProp         = gather->getAvailableMemoryProportion();
  const int64_t virtualGraphCount = op->getIr().getDeviceInfo()->getNumIpus();

  const auto numElements = std::accumulate(
      inputShape.begin(), inputShape.end(), 1, std::multiplies<int64_t>());

  const auto split =
      boost::integer::gcd(virtualGraphCount, numElements / inputShape[axis]);
  const auto stride      = (numElements / inputShape[axis]) / split;
  const auto inTensorId  = op->input->id(GatherOp::dataInIndex());
  const auto idxTensorId = op->input->id(GatherOp::indicesInIndex());
  auto output            = op->outTensor(GatherOp::outIndex());
  op->disconnectOutTensor(output);
  const auto outTensorId = output->id;

  logging::pattern::trace("Splitting {} into {} gathers of size [{}, {}]",
                          op->str(),
                          split,
                          inputShape[axis],
                          (numElements / inputShape[axis]) / split);

  auto t0        = inTensorId;
  auto transpose = canonicalizeTranspose(axis, inputShape, op->settings);
  auto t1        = output->getIr().createIntermediateTensorId(outTensorId);
  transpose->connectInTensor(0, t0);
  transpose->createAndConnectOutTensor(0, t1);
  transpose->setup();

  // Reshape the input tensor to a 2D tensor with the gather axis at the front
  auto reshape = canonicalizeShape(axis, inputShape, op->settings);
  auto t2      = transpose->getIr().createIntermediateTensorId(outTensorId);
  reshape->connectInTensor(0, t1);
  reshape->createAndConnectOutTensor(0, t2);
  reshape->setup();

  // Slice the input tensor along the sequence dimension for each IPU
  auto slices = createSlices(split, stride, op->settings);
  std::vector<TensorId> slicedts;
  slicedts.reserve(split);
  for (int i = 0; i < split; ++i) {
    auto tid = reshape->getIr().createIntermediateTensorId(outTensorId);
    slicedts.push_back(tid);
    slices[i]->connectInTensor(0, t2);
    slices[i]->createAndConnectOutTensor(0, tid);
    slices[i]->setup();
  }

  // Gather the slice fragments on each IPU
  auto gathers = createGathers(split, op->settings, availMemProp);
  std::vector<TensorId> gatheredts;
  gatheredts.reserve(split);
  for (int i = 0; i < split; ++i) {
    auto tid = gathers[i]->getIr().createIntermediateTensorId(outTensorId);
    gatheredts.push_back(tid);
    gathers[i]->connectInTensor(0, slicedts[i]);
    gathers[i]->connectInTensor(1, idxTensorId);
    gathers[i]->createAndConnectOutTensor(0, tid);
    gathers[i]->setup();
  }

  // Concatenate the gathered fragments
  const auto concatDim = gather->inRank(1);
  auto concat          = std::make_unique<ConcatOp>(
      Onnx::Operators::Concat_4, concatDim, op->settings);
  auto t3 = concat->getIr().createIntermediateTensorId(outTensorId);
  for (int i = 0; i < split; ++i) {
    concat->connectInTensor(i, gatheredts[i]);
  }
  concat->createAndConnectOutTensor(0, t3);
  concat->setup();

  const auto indicesShape = gather->inShape(1);

  // Reshape back to the original shape, with the gather axis at the front
  auto unreshape =
      decanonicalizeShape(axis, inputShape, indicesShape, op->settings);
  auto t4 = concat->getIr().createIntermediateTensorId(outTensorId);
  unreshape->connectInTensor(0, t3);
  unreshape->createAndConnectOutTensor(0, t4);
  unreshape->setup();

  // Put the gather axis back in original position
  auto untranspose =
      decanonicalizeTranspose(axis, inputShape, indicesShape, op->settings);
  auto t5 = outTensorId;
  untranspose->connectInTensor(0, t4);
  untranspose->connectOutTensor(0, t5);
  untranspose->setup();

  // Insert the ops into the IR
  auto &graph = op->getGraph();
  graph.moveIntoGraph(std::move(transpose));
  graph.moveIntoGraph(std::move(reshape));
  for (auto &s : slices) {
    graph.moveIntoGraph(std::move(s));
  }
  for (auto &g : gathers) {
    graph.moveIntoGraph(std::move(g));
  }
  graph.moveIntoGraph(std::move(concat));
  graph.moveIntoGraph(std::move(unreshape));
  graph.moveIntoGraph(std::move(untranspose));

  // Remove the old op
  op->disconnectAllInputs();
  op->getGraph().eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<SplitGatherPattern> splitGatherer("SplitGather", false);
}

} // namespace popart
