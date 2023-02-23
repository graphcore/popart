// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/Fill.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/op/scatterreduce.hpp>
#include <popart/popx/op/scatterreduceutilx.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/sliceplanx.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/vendored/optional.hpp"

namespace pe = popops::expr;

namespace popart {
namespace popx {

namespace {

template <typename FuncT, typename... Args>
static decltype(auto) conditionalCall(bool condition,
                                      FuncT funcTrue,
                                      FuncT funcFalse,
                                      Args &&... args) {

  static_assert(std::is_function_v<typename std::remove_pointer_t<FuncT>>);

  if (condition)
    return funcTrue(std::forward<Args>(args)...);

  return funcFalse(std::forward<Args>(args)...);
}

static std::string prepareDebugName(const char *name, bool isGrouped) {
  std::string debugName = name;
  if (isGrouped) {
    debugName[0] = toupper(debugName[0]);
    debugName    = "grouped" + debugName;
  }

  return debugName;
}

template <typename FuncT, typename... Args>
static void multiUpdate(const Opx &opx,
                        const char *name,
                        bool isGrouped,
                        FuncT groupedMultiUpdateFunc,
                        FuncT multiUpdateFunc,
                        Args &&... args) {

  const std::string debugName = prepareDebugName(name, isGrouped);
  conditionalCall(isGrouped,
                  groupedMultiUpdateFunc,
                  multiUpdateFunc,
                  std::forward<Args>(args)...,
                  opx.debugContext(debugName));
}

template <typename FuncT, typename... Args>
static decltype(auto) multiSlice(const Opx &opx,
                                 const char *name,
                                 bool isGrouped,
                                 FuncT groupedMultiSliceFunc,
                                 FuncT multiSliceFunc,
                                 Args &&... args) {

  const std::string debugName = prepareDebugName(name, isGrouped);
  return conditionalCall(isGrouped,
                         groupedMultiSliceFunc,
                         multiSliceFunc,
                         std::forward<Args>(args)...,
                         opx.debugContext(debugName));
}

static void maskedFillOutput(const Opx &opx,
                             const poplar::Tensor &out,
                             const poplar::Tensor &src,
                             const poplar::Tensor &indices,
                             poplar::program::Sequence &seq,
                             const popops::SlicePlan &plan,
                             const bool isGrouped) {
  auto &graph = opx.graph();

  const auto mask = opx.cloneNcopy(seq, out);
  popops::zero(graph, mask, seq, "zeros");

  const auto ones = opx.cloneNcopy(seq, src);
  popops::fill(graph, ones, seq, 1, "ones");

  popx::multiUpdate(opx,
                    "scatterMask",
                    isGrouped,
                    popops::groupedMultiUpdateMax,
                    popops::multiUpdateMax,
                    graph,
                    mask,
                    ones,
                    indices,
                    std::vector<std::size_t>{0},
                    std::vector<std::size_t>{1},
                    seq,
                    plan,
                    poplar::OptionFlags());

  const auto expr = pe::Select(pe::_1, pe::_2, pe::Cast(pe::_2, poplar::BOOL));
  popops::mapInPlace(
      graph, expr, {out, mask}, seq, opx.debugContext("maskedFill"));
}

template <typename T>
static void initOutTensor(const Opx &opx,
                          const poplar::Tensor &out,
                          poplar::program::Sequence &prog,
                          const T value) {
  popops::fill<T>(opx.graph(), out, prog, value, opx.debugContext("initOut"));
}

} // namespace

namespace scatterreduceutilx {

void ScatterReductionStrategy::initReductionOutput(
    const Opx &opx,
    const poplar::Tensor &out,
    poplar::program::Sequence &prog) const {
  popops::zero(opx.graph(), out, prog, opx.debugContext("zero"));
}

void ScatterReductionStrategy::forward(const ScatterReduceOp &op,
                                       const Opx &opx,
                                       poplar::Tensor output,
                                       poplar::Tensor data,
                                       poplar::Tensor indices,
                                       size_t axis,
                                       size_t group_size,
                                       poplar::program::Sequence &prog,
                                       const popops::SlicePlan &plan) const {
  const bool isGrouped = group_size > 1;
  if (op.indexBroadcasted() || !op.indexBroadcastEnabled()) {
    prepMultiUpdateBroadcastedTensors(
        opx, output, data, indices, axis, group_size, prog);
  } else {
    prepMultiUpdateTensors(output, data, indices, axis, group_size);
  }

  // Delegate to concrete subclasses the evaluation of the reduction.
  applyReduction(opx, output, data, indices, prog, plan, isGrouped);
}

std::vector<poplar::Tensor>
ScatterReductionStrategy::backward(const ScatterReduceGradOp &op,
                                   const Opx &opx,
                                   poplar::Tensor &gradIn,
                                   poplar::Tensor &indices,
                                   size_t axis,
                                   size_t group_size,
                                   poplar::program::Sequence &prog,
                                   const popops::SlicePlan &plan) const {
  if (op.indexBroadcasted() || !op.indexBroadcastEnabled()) {
    prepMultiSliceBroadcastedTensors(
        opx, gradIn, indices, axis, group_size, prog);
  } else {
    prepMultiSliceTensors(gradIn, indices, axis, group_size);
  }

  // Delegate to concrete subclasses the evaluation of the gradient(s).
  return calcGradient(op, opx, gradIn, indices, axis, group_size, prog, plan);
}

poplar::OptionFlags ScatterReductionStrategy::createForwardPlanOptions(
    const ScatterReduceOp &op) const {
  return createSlicePlanOptions(forwardSlicePlanUsedFor(),
                                op.getAvailableMemoryProportion());
}

poplar::OptionFlags ScatterReductionStrategy::createBackwardPlanOptions(
    const ScatterReduceGradOp &op) const {
  return createSlicePlanOptions(SlicePlanUsedFor::Slice,
                                op.getAvailableMemoryProportion());
}

void ScatterReductionStrategy::prepMultiUpdateBroadcastedTensors(
    const Opx &opx,
    poplar::Tensor &output,
    poplar::Tensor &data,
    poplar::Tensor &indices,
    size_t axis,
    size_t group_size,
    poplar::program::Sequence &prog) const {
  // The popops::multiUpdateAdd op is roughly:
  //   for i indices:
  //    out[indices[i]] += data[i]
  // but the output must be 2d. To support inputs with rank > 2 we do:
  //   * permute dims of data and indices and output so that slice axis == 0
  //   * indices are linearized into a 1-d coordinate system
  //   * flatten the remaining dims
  const auto indicesRank = indices.rank();

  if (indicesRank != data.rank()) {
    // Expect this to be handled in ScatterReduceOp::checkIndexBroadcasted
    throw error("Partial broadcasting of indices is not currently supported.");
  }
  const bool isGrouped        = group_size > 1;
  const unsigned startAxisDim = isGrouped ? 1 : 0;

  const auto &outputShape = output.shape();
  const auto &dataShape   = data.shape();
  const auto &indicesShape =
      expandIndicesBcastShape(indices.shape(), dataShape, axis, isGrouped);

  if (shouldShrinkTensor(dataShape, indicesShape))
    data = shrinkTensorToFitShape(data, indicesShape);

  if (shouldShrinkTensor(outputShape, indicesShape, axis))
    output = shrinkTensorToFitShape(output, indicesShape, axis);

  output  = output.dimRoll(axis, startAxisDim);
  data    = data.dimRoll(axis, startAxisDim);
  indices = indices.dimRoll(axis, startAxisDim);

  if (indicesRank < 2) {
    output  = output.expand({1});
    indices = indices.expand({1});
    data    = data.expand({1, 1});
  } else {
    const int numDataCols = calcNumDataCols(output, startAxisDim);
    indices               = scatterutilx::linearizeIndices(
        opx, prog, indices, numDataCols, group_size);

    output = output.flatten();
    output = output.expand({1});
    if (isGrouped)
      output =
          output.reshapePartial(0, 1, {group_size, output.dim(0) / group_size});

    data = data.flatten(isGrouped ? 2 : 1, data.rank());
    data = data.flatten();
    data = data.expand({1, 1});
    if (isGrouped)
      data = data.reshapePartial(0, 1, {group_size, data.dim(0) / group_size});
  }

  // Assume indices are non-negative
  indices = indices.reinterpret(poplar::UNSIGNED_INT);
}

void ScatterReductionStrategy::prepMultiUpdateTensors(poplar::Tensor &output,
                                                      poplar::Tensor &data,
                                                      poplar::Tensor &indices,
                                                      size_t axis,
                                                      size_t group_size) const {
  const bool isGrouped        = group_size > 1;
  const unsigned startAxisDim = isGrouped ? 1 : 0;

  const auto dataShape = data.shape();

  if (shouldShrinkTensor(output.shape(), dataShape, axis))
    output = shrinkTensorToFitShape(output, dataShape, axis);

  // Place the reduction axis at the front
  output = output.dimRoll(axis, startAxisDim);
  data   = data.dimRoll(axis, startAxisDim);

  // flatten the remaining dims to handle rank > 2 inputs
  output = output.flatten(startAxisDim + 1, output.rank());
  data   = data.flatten(startAxisDim + 1, data.rank());

  // Add a trailing singleton dimension to the data input for slicing.
  data = data.expand({startAxisDim + 1});

  // Indices should be a vector but may contain some singleton dimensions.
  indices = indices.flatten(startAxisDim, indices.rank());
  indices = indices.expand({startAxisDim + 1});
  // Assume indices are non-negative
  indices = indices.reinterpret(poplar::UNSIGNED_INT);
}

void ScatterReductionStrategy::prepMultiSliceBroadcastedTensors(
    const Opx &opx,
    poplar::Tensor &data,
    poplar::Tensor &indices,
    size_t axis,
    size_t group_size,
    poplar::program::Sequence &prog) const {
  // The gradient of a scatter reduction requires a gather (multiSlice).
  // Just like in the forward pass we need to support inputs with rank > 2 so:
  //   * permute dims of data and indices and output so that slice axis == 0
  //   * indices are linearized into a 1-d coordinate system
  //   * flatten the remaining dims
  const bool isGrouped        = group_size > 1;
  const unsigned startAxisDim = isGrouped ? 1 : 0;
  data                        = data.dimRoll(axis, startAxisDim);
  indices                     = indices.dimRoll(axis, startAxisDim);

  if (indices.rank() < 2) {
    data    = data.expand({1});
    indices = indices.expand({1});
  } else {
    const auto numCols =
        (indices.numElements() / indices.shape().at(startAxisDim)) / group_size;
    indices =
        scatterutilx::linearizeIndices(opx, prog, indices, numCols, group_size);
    data = data.flatten();
    data = data.expand({1});
    if (isGrouped)
      data = data.reshapePartial(0, 1, {group_size, data.dim(0) / group_size});
  }

  // Assume indices are non-negative
  indices = indices.reinterpret(poplar::UNSIGNED_INT);
}

void ScatterReductionStrategy::prepMultiSliceTensors(poplar::Tensor &data,
                                                     poplar::Tensor &indices,
                                                     size_t axis,
                                                     size_t group_size) const {
  const bool isGrouped        = group_size > 1;
  const unsigned startAxisDim = isGrouped ? 1 : 0;
  // Place the gather axis at the front.

  data = data.dimRoll(axis, startAxisDim);
  data = data.flatten(startAxisDim + 1, data.rank());

  // Indices should be a vector but may contain some singleton dimensions.
  indices = indices.flatten();
  indices = indices.expand({1});
  if (isGrouped)
    indices =
        indices.reshapePartial(0, 1, {group_size, indices.dim(0) / group_size});
  // Assume indices are non-negative
  indices = indices.reinterpret(poplar::UNSIGNED_INT);
}

bool ScatterReductionStrategy::shouldShrinkTensor(
    const std::vector<std::size_t> &tensorShape,
    const std::vector<std::size_t> &shrinkShape,
    const nonstd::optional<std::size_t> skipAxis) const {
  const auto rank = shrinkShape.size();

  for (std::size_t i = 0; i < rank; ++i) {
    if (i == skipAxis)
      continue;

    if (shrinkShape.at(i) < tensorShape.at(i))
      return true;
  }
  return false;
}

poplar::Tensor ScatterReductionStrategy::shrinkTensorToFitShape(
    const poplar::Tensor &tensor,
    const std::vector<std::size_t> &shape,
    const nonstd::optional<std::size_t> skipAxis) const {

  const auto rank = shape.size();
  const std::vector<std::size_t> begin(rank, 0);
  std::vector<std::size_t> end = tensor.shape();

  for (std::size_t i = 0; i < rank; ++i) {
    if (i == skipAxis)
      continue;

    if (const auto idim = shape.at(i); idim < tensor.dim(i)) {
      end[i] = idim;
    }
  }
  return tensor.slice(begin, end);
}

int ScatterReductionStrategy::calcNumDataCols(const poplar::Tensor &tensor,
                                              const int startDim) const {
  const auto shape = tensor.shape();

  return std::accumulate(
      shape.cbegin() + startDim + 1, shape.cend(), 1, std::multiplies<int>());
}

template <typename FuncT, typename... Args>
static decltype(auto) conditionalCall(bool condition,
                                      FuncT funcTrue,
                                      FuncT funcFalse,
                                      Args &&... args) {

  static_assert(std::is_function_v<typename std::remove_pointer_t<FuncT>>);

  if (condition)
    return funcTrue(std::forward<Args>(args)...);

  return funcFalse(std::forward<Args>(args)...);
}

SlicePlanUsedFor SumReductionStrategy::forwardSlicePlanUsedFor() const {
  return SlicePlanUsedFor::UpdateAdd;
}

void SumReductionStrategy::applyReduction(const Opx &opx,
                                          const poplar::Tensor &target,
                                          const poplar::Tensor &update,
                                          const poplar::Tensor &indices,
                                          poplar::program::Sequence &prog,
                                          const popops::SlicePlan &plan,
                                          const bool isGrouped) const {
  auto &graph      = opx.graph();
  const auto scale = graph.addConstant(
      update.elementType(), {}, 1.0f, opx.debugContext("const_1"));
  graph.setTileMapping(scale, 0);

  popx::multiUpdate(opx,
                    "scatterSum",
                    isGrouped,
                    popops::groupedMultiUpdateAdd,
                    popops::multiUpdateAdd,
                    graph,
                    target,
                    update,
                    indices,
                    scale,
                    std::vector<std::size_t>{0},
                    std::vector<std::size_t>{1},
                    prog,
                    plan,
                    poplar::OptionFlags());
}

std::vector<poplar::Tensor>
SumReductionStrategy::calcGradient(const ScatterReduceGradOp &op,
                                   const Opx &opx,
                                   const poplar::Tensor &gradIn,
                                   const poplar::Tensor &indices,
                                   size_t axis,
                                   size_t group_size,
                                   poplar::program::Sequence &prog,
                                   const popops::SlicePlan &plan) const {
  auto gradData = popx::multiSlice(opx,
                                   "scatterSumGrad",
                                   group_size > 1,
                                   popops::groupedMultiSlice,
                                   popops::multiSlice,
                                   opx.graph(),
                                   gradIn,
                                   indices,
                                   std::vector<std::size_t>{0},
                                   std::vector<std::size_t>{1},
                                   prog,
                                   plan,
                                   poplar::OptionFlags());

  auto outInfo = opx.outInfo(ScatterReduceGradOp::gradDataOutIndex());
  gradData     = alignToAxis(gradData, outInfo.shape(), axis, group_size);

  if (op.hasInitialValues()) {
    const auto gradInitials = opx.cloneNcopy(
        prog, opx.getInTensor(ScatterReduceGradOp::gradInIndex()));
    return {gradData, gradInitials};
  }

  return {gradData};
}

SlicePlanUsedFor NoneReductionStrategy::forwardSlicePlanUsedFor() const {
  return SlicePlanUsedFor::Update;
}

poplar::OptionFlags NoneReductionStrategy::createBackwardPlanOptions(
    const ScatterReduceGradOp &op) const {
  if (op.hasInitialValues()) {
    // none-reduction with initial values needs both a scatter and a gather
    // for each gradient output.
    return createSlicePlanOptions(SlicePlanUsedFor::CombinedSliceUpdate,
                                  op.getAvailableMemoryProportion());
  }

  return createSlicePlanOptions(SlicePlanUsedFor::Slice,
                                op.getAvailableMemoryProportion());
}

void NoneReductionStrategy::applyReduction(const Opx &opx,
                                           const poplar::Tensor &target,
                                           const poplar::Tensor &update,
                                           const poplar::Tensor &indices,
                                           poplar::program::Sequence &prog,
                                           const popops::SlicePlan &plan,
                                           const bool isGrouped) const {

  popx::multiUpdate(opx,
                    "scatter",
                    isGrouped,
                    popops::groupedMultiUpdate,
                    popops::multiUpdate,
                    opx.graph(),
                    target,
                    update,
                    indices,
                    std::vector<std::size_t>{0},
                    std::vector<std::size_t>{1},
                    prog,
                    plan,
                    poplar::OptionFlags());
}

std::vector<poplar::Tensor>
NoneReductionStrategy::calcGradient(const ScatterReduceGradOp &op,
                                    const Opx &opx,
                                    const poplar::Tensor &gradIn,
                                    const poplar::Tensor &indices,
                                    size_t axis,
                                    size_t group_size,
                                    poplar::program::Sequence &prog,
                                    const popops::SlicePlan &plan) const {
  const bool isGrouped = group_size > 1;

  auto gradData = popx::multiSlice(opx,
                                   "scatterNoneGrad",
                                   isGrouped,
                                   popops::groupedMultiSlice,
                                   popops::multiSlice,
                                   opx.graph(),
                                   gradIn,
                                   indices,
                                   std::vector<std::size_t>{0},
                                   std::vector<std::size_t>{1},
                                   prog,
                                   plan,
                                   poplar::OptionFlags());

  const auto gradDataOutInfo =
      opx.outInfo(ScatterReduceGradOp::gradDataOutIndex());
  gradData = alignToAxis(gradData, gradDataOutInfo.shape(), axis, group_size);

  if (!op.hasInitialValues()) {
    return {gradData};
  }

  // Scatter of zeros into the gradIn
  const auto dataInfo    = opx.inInfo(ScatterReduceGradOp::dataInIndex());
  const auto indicesInfo = opx.inInfo(ScatterReduceGradOp::indicesInIndex());

  auto numSlices  = static_cast<size_t>(dataInfo.nelms()) / group_size;
  auto outputSize = 1UL;

  if (!op.indexBroadcasted()) {
    numSlices  = dataInfo.shape_szt().at(axis);
    outputSize = (dataInfo.nelms() / numSlices) / group_size;
  }

  const auto numLookups = static_cast<size_t>(indicesInfo.nelms()) / group_size;

  auto zeros =
      isGrouped
          ? popops::createGroupedSliceTensor(
                opx.graph(),
                gradIn.elementType(),
                group_size,
                {numSlices, outputSize},
                {0},
                {1},
                numLookups,
                plan,
                poplar::OptionFlags(),
                opx.getDebugNameAndId("groupedZerosUpdate"))
          : popops::createSliceTensor(opx.graph(),
                                      gradIn.elementType(),
                                      {numSlices, outputSize},
                                      {0},
                                      {1},
                                      numLookups,
                                      plan,
                                      poplar::OptionFlags(),
                                      opx.getDebugNameAndId("zerosUpdate"));

  popops::fill(
      opx.graph(), zeros, prog, 0.0f, opx.debugContext("zerosUpdateFill"));

  auto gradInitials =
      isGrouped
          ? popops::createGroupedSliceableTensor(
                opx.graph(),
                gradIn.elementType(),
                group_size,
                {gradIn.dim(1), gradIn.dim(2)},
                {0},
                {1},
                plan,
                poplar::OptionFlags(),
                opx.getDebugNameAndId("groupedZerosUpdate"))
          : popops::createSliceableTensor(opx.graph(),
                                          gradIn.elementType(),
                                          {gradIn.dim(0), gradIn.dim(1)},
                                          {0},
                                          {1},
                                          plan,
                                          poplar::OptionFlags(),
                                          opx.getDebugNameAndId("zerosUpdate"));

  prog.add(poplar::program::Copy(
      gradIn, gradInitials, false, opx.debugContext("copyToScatter")));

  popx::multiUpdate(opx,
                    "scatterInitialValuesGrad",
                    isGrouped,
                    popops::groupedMultiUpdate,
                    popops::multiUpdate,
                    opx.graph(),
                    gradInitials,
                    zeros,
                    indices,
                    std::vector<std::size_t>{0},
                    std::vector<std::size_t>{1},
                    prog,
                    plan,
                    poplar::OptionFlags());

  const auto info =
      opx.outInfo(ScatterReduceGradOp::gradInitialValuesOutIndex());
  gradInitials = alignToAxis(gradInitials, info.shape(), axis, group_size);
  return {gradData, gradInitials};
}

SlicePlanUsedFor MaxReductionStrategy::forwardSlicePlanUsedFor() const {
  return SlicePlanUsedFor::UpdateMax;
}

namespace {

template <typename T> static T maxInitValue() {
  return std::numeric_limits<T>::has_infinity
             ? -std::numeric_limits<T>::infinity()
             : std::numeric_limits<T>::lowest();
}

} // namespace

void MaxReductionStrategy::initReductionOutput(
    const Opx &opx,
    const poplar::Tensor &out,
    poplar::program::Sequence &prog) const {
  const auto dtype = out.elementType();

  if (dtype == poplar::FLOAT || dtype == poplar::HALF) {
    initOutTensor(opx, out, prog, maxInitValue<float>());
  } else if (dtype == poplar::INT) {
    initOutTensor(opx, out, prog, maxInitValue<int>());
  } else if (dtype == poplar::UNSIGNED_INT) {
    initOutTensor(opx, out, prog, maxInitValue<unsigned int>());
  } else {
    throw popart::internal_error("Unsupported data type {}", dtype);
  }
}

void MaxReductionStrategy::applyReduction(const Opx &opx,
                                          const poplar::Tensor &target,
                                          const poplar::Tensor &update,
                                          const poplar::Tensor &indices,
                                          poplar::program::Sequence &prog,
                                          const popops::SlicePlan &plan,
                                          const bool isGrouped) const {

  popx::multiUpdate(opx,
                    "scatterMax",
                    isGrouped,
                    popops::groupedMultiUpdateMax,
                    popops::multiUpdateMax,
                    opx.graph(),
                    target,
                    update,
                    indices,
                    std::vector<std::size_t>{0},
                    std::vector<std::size_t>{1},
                    prog,
                    plan,
                    poplar::OptionFlags());

  if (!opx.hasInput(ScatterReduceOp::initialValuesInIndex())) {
    // TODO(T65173): make this an operator option since it can be unnecessary.
    // Replace any non-updated values with zero
    maskedFillOutput(opx, target, update, indices, prog, plan, isGrouped);
  }
}

std::vector<poplar::Tensor>
MaxReductionStrategy::calcGradient(const ScatterReduceGradOp &op,
                                   const Opx &opx,
                                   const poplar::Tensor &gradIn,
                                   const poplar::Tensor &indices,
                                   size_t axis,
                                   size_t group_size,
                                   poplar::program::Sequence &prog,
                                   const popops::SlicePlan &plan) const {
  const bool isGrouped = group_size > 1;
  auto &graph          = opx.graph();
  auto gradData        = popx::multiSlice(opx,
                                   "gatherGradIn",
                                   isGrouped,
                                   popops::groupedMultiSlice,
                                   popops::multiSlice,
                                   graph,
                                   gradIn,
                                   indices,
                                   std::vector<std::size_t>{0},
                                   std::vector<std::size_t>{1},
                                   prog,
                                   plan,
                                   poplar::OptionFlags());

  const auto gradDataInfo =
      opx.outInfo(ScatterReduceGradOp::gradDataOutIndex());
  gradData = alignToAxis(gradData, gradDataInfo.shape(), axis, group_size);

  auto fwdOut = opx.getInTensor(ScatterReduceGradOp::fwdOutInIndex());

  const unsigned startAxisDim = isGrouped ? 1 : 0;
  if (op.indexBroadcasted()) {
    fwdOut = fwdOut.dimRoll(axis, startAxisDim);
    fwdOut = fwdOut.flatten();
    fwdOut = fwdOut.expand({1});
    if (isGrouped)
      fwdOut =
          fwdOut.reshapePartial(0, 1, {group_size, fwdOut.dim(0) / group_size});
  } else {
    fwdOut = fwdOut.dimRoll(axis, startAxisDim);
    fwdOut = fwdOut.flatten(1 + startAxisDim, fwdOut.rank());
  }

  auto outputs = popx::multiSlice(opx,
                                  "gatherFwdOut",
                                  isGrouped,
                                  popops::groupedMultiSlice,
                                  popops::multiSlice,
                                  graph,
                                  fwdOut,
                                  indices,
                                  std::vector<std::size_t>{0},
                                  std::vector<std::size_t>{1},
                                  prog,
                                  plan,
                                  poplar::OptionFlags());
  outputs      = alignToAxis(outputs, gradDataInfo.shape(), axis, group_size);

  // gradDataOut * (data == outputs)
  const auto &data = opx.getInTensor(ScatterReduceGradOp::dataInIndex());
  const auto dtype = gradData.elementType();
  const auto expr = pe::Mul(pe::_1, pe::Cast(pe::Equal(pe::_2, pe::_3), dtype));

  popops::mapInPlace(opx.graph(),
                     expr,
                     {gradData, data, outputs},
                     prog,
                     opx.getDebugNameAndId("gradOutMulOutputMask"));

  if (!op.hasInitialValues()) {
    return {gradData};
  }

  // gradIn * (fwdOut == initialValues)
  const auto grad = opx.getInTensor(ScatterReduceGradOp::gradInIndex());
  const auto fwd  = opx.getInTensor(ScatterReduceGradOp::fwdOutInIndex());
  const auto initialValues =
      opx.getInTensor(ScatterReduceGradOp::initialValuesInIndex());

  const auto gradInitials =
      popops::map(opx.graph(),
                  expr,
                  {grad, fwd, initialValues},
                  prog,
                  opx.getDebugNameAndId("gradInMulInitialValuesMask"));

  return {gradData, gradInitials};
}

void MinReductionStrategy::applyReduction(const Opx &opx,
                                          const poplar::Tensor &target,
                                          const poplar::Tensor &update,
                                          const poplar::Tensor &indices,
                                          poplar::program::Sequence &prog,
                                          const popops::SlicePlan &plan,
                                          const bool isGrouped) const {
  auto &graph = opx.graph();

  if (opx.hasInput(ScatterReduceOp::initialValuesInIndex())) {
    popops::negInPlace(graph, target, prog, "negTarget");
  }

  const auto negUpdate = popops::neg(graph, update, prog, "negUpdate");
  MaxReductionStrategy::applyReduction(
      opx, target, negUpdate, indices, prog, plan, isGrouped);
  popops::negInPlace(graph, target, prog, "negTarget");
}

SlicePlanUsedFor MulReductionStrategy::forwardSlicePlanUsedFor() const {
  return SlicePlanUsedFor::UpdateMul;
}

void MulReductionStrategy::initReductionOutput(
    const Opx &opx,
    const poplar::Tensor &out,
    poplar::program::Sequence &prog) const {
  const auto dtype = out.elementType();

  if (dtype == poplar::FLOAT || dtype == poplar::HALF) {
    initOutTensor(opx, out, prog, 1.0f);
  } else if (dtype == poplar::INT) {
    initOutTensor(opx, out, prog, 1);
  } else if (dtype == poplar::UNSIGNED_INT) {
    initOutTensor(opx, out, prog, 1u);
  } else {
    throw popart::internal_error("Unsupported data type {}", dtype);
  }
}

void MulReductionStrategy::applyReduction(const Opx &opx,
                                          const poplar::Tensor &target,
                                          const poplar::Tensor &update,
                                          const poplar::Tensor &indices,
                                          poplar::program::Sequence &prog,
                                          const popops::SlicePlan &plan,
                                          const bool isGrouped) const {

  popx::multiUpdate(opx,
                    "scatterMul",
                    isGrouped,
                    popops::groupedMultiUpdateMul,
                    popops::multiUpdateMul,
                    opx.graph(),
                    target,
                    update,
                    indices,
                    std::vector<std::size_t>{0},
                    std::vector<std::size_t>{1},
                    prog,
                    plan,
                    poplar::OptionFlags());
}

std::vector<poplar::Tensor>
MulReductionStrategy::calcGradient(const ScatterReduceGradOp &op,
                                   const Opx &opx,
                                   const poplar::Tensor &gradIn,
                                   const poplar::Tensor &indices,
                                   size_t axis,
                                   size_t group_size,
                                   poplar::program::Sequence &prog,
                                   const popops::SlicePlan &plan) const {
  const bool isGrouped = group_size > 1;
  auto &graph          = opx.graph();

  auto gradData = popx::multiSlice(opx,
                                   "gatherGradIn",
                                   isGrouped,
                                   popops::groupedMultiSlice,
                                   popops::multiSlice,
                                   graph,
                                   gradIn,
                                   indices,
                                   std::vector<std::size_t>{0},
                                   std::vector<std::size_t>{1},
                                   prog,
                                   plan,
                                   poplar::OptionFlags());
  const auto gradDataInfo =
      opx.outInfo(ScatterReduceGradOp::gradDataOutIndex());
  gradData = alignToAxis(gradData, gradDataInfo.shape(), axis, group_size);

  auto fwdOut = opx.getInTensor(ScatterReduceGradOp::fwdOutInIndex());
  const unsigned startAxisDim = isGrouped ? 1 : 0;
  if (op.indexBroadcasted()) {
    fwdOut = fwdOut.dimRoll(axis, startAxisDim);
    fwdOut = fwdOut.flatten();
    fwdOut = fwdOut.expand({1});
    if (isGrouped)
      fwdOut =
          fwdOut.reshapePartial(0, 1, {group_size, fwdOut.dim(0) / group_size});
  } else {
    fwdOut = fwdOut.dimRoll(axis, startAxisDim);
    fwdOut = fwdOut.flatten(1 + startAxisDim, fwdOut.rank());
  }

  auto outputs = popx::multiSlice(opx,
                                  "gatherFwdOut",
                                  isGrouped,
                                  popops::groupedMultiSlice,
                                  popops::multiSlice,
                                  graph,
                                  fwdOut,
                                  indices,
                                  std::vector<std::size_t>{0},
                                  std::vector<std::size_t>{1},
                                  prog,
                                  plan,
                                  poplar::OptionFlags());
  outputs      = alignToAxis(outputs, gradDataInfo.shape(), axis, group_size);

  // gradDataOut * (fwdOutputs / data)
  const auto &data = opx.getInTensor(ScatterReduceGradOp::dataInIndex());
  const auto dtype = gradData.elementType();
  const auto expr =
      pe::Mul(pe::_1, pe::Mul(pe::Cast(pe::_2, dtype), pe::Inv(pe::_3)));

  popops::mapInPlace(opx.graph(),
                     expr,
                     {gradData, outputs, data},
                     prog,
                     opx.getDebugNameAndId("gradOutMulOutputMask"));

  if (!op.hasInitialValues()) {
    return {gradData};
  }

  // gradDataOut * (fwdOutputs / data)
  const auto grad = opx.getInTensor(ScatterReduceGradOp::gradInIndex());
  const auto fwd  = opx.getInTensor(ScatterReduceGradOp::fwdOutInIndex());
  const auto initialValues =
      opx.getInTensor(ScatterReduceGradOp::initialValuesInIndex());

  const auto gradInitials =
      popops::map(opx.graph(),
                  expr,
                  {grad, fwd, initialValues},
                  prog,
                  opx.getDebugNameAndId("gradInMulInitialValuesMask"));

  return {gradData, gradInitials};
}

} // namespace scatterreduceutilx
} // namespace popx
} // namespace popart
