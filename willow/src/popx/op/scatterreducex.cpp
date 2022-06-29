// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <cstddef>
#include <limits>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <snap/popops/ElementWise.hpp>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Type.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Fill.hpp>
#include <popops/Zero.hpp>
#include <popart/error.hpp>
#include <popart/op/scatterreduce.hpp>
#include <popart/popx/op/scatterreducex.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/popx/op/gatherx.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/tensorinfo.hpp"

namespace pe = popops::expr;

namespace popart {
class Op;

namespace popx {
class Devicex;

/**
 * Strategy implementation for calculating the forward and backward passes of
 * the scatter reduction.
 *
 * Uses the template pattern to delegate the details of calling the appropriate
 * poplibs API for the requested reduction mode. This interface will handles the
 * preparation of the broadcasted vs vectorised cases for the index input.
 */
class ReductionStrategy {
public:
  virtual poplar::OptionFlags
  createSlicePlanOptions(const ScatterReduceOp &op) const = 0;

  virtual void initReductionOutput(const PopOpx &opx,
                                   const snap::Tensor &out,
                                   snap::program::Sequence &prog) const = 0;

  virtual void applyReduction(const PopOpx &opx,
                              const snap::Tensor &target,
                              const snap::Tensor &update,
                              const snap::Tensor &indices,
                              snap::program::Sequence &prog,
                              const popops::SlicePlan &plan) const = 0;

  void forward(const ScatterReduceOp &op,
               const PopOpx &opx,
               snap::Tensor output,
               snap::Tensor data,
               snap::Tensor indices,
               size_t axis,
               snap::program::Sequence &prog,
               const popops::SlicePlan &plan) const {
    if (op.indexBroadcasted()) {
      prepMultiUpdateBroadcastedTensors(opx, output, data, indices, axis, prog);
    } else {
      prepMultiUpdateTensors(output, data, indices, axis);
    }

    // Delegate to concrete subclasses the evaluation of the reduction.
    applyReduction(opx, output, data, indices, prog, plan);
  }

  void prepMultiUpdateBroadcastedTensors(const PopOpx &opx,
                                         snap::Tensor &output,
                                         snap::Tensor &data,
                                         snap::Tensor &indices,
                                         size_t axis,
                                         snap::program::Sequence &prog) const {
    // The popops::multiUpdateAdd op is roughly:
    //   for i indices:
    //    out[indices[i]] += data[i]
    // but the output must be 2d. To support inputs with rank > 2 we do:
    //   * permute dims of data and indices and output so that slice axis == 0
    //   * indices are linearized into a 1-d coordinate system
    //   * flatten the remaining dims
    if (indices.rank() != data.rank()) {
      // Expect this to be handled in ScatterReduceOp::checkIndexBroadcasted
      throw error(
          "Partial broadcasting of indices is not currently supported.");
    }
    output  = output.dimRoll(axis);
    data    = data.dimRoll(axis);
    indices = indices.dimRoll(axis);

    if (indices.rank() < 2) {
      output  = output.expand({1});
      indices = indices.expand({1});
      data    = data.expand({1, 1});
    } else {
      output           = output.flatten();
      output           = output.expand({1});
      data             = data.flatten(1, data.rank());
      auto numDataCols = static_cast<int>(data.dim(1));
      indices = scatterutilx::linearizeIndices(opx, prog, indices, numDataCols);
      data    = data.flatten();
      data    = data.expand({1, 1});
    }

    // Assume indices are non-negative
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
  }

  void prepMultiUpdateTensors(snap::Tensor &output,
                              snap::Tensor &data,
                              snap::Tensor &indices,
                              size_t axis) const {
    // Place the reduction axis at the front
    output = output.dimRoll(axis);
    data   = data.dimRoll(axis);

    // flatten the remaining dims to handle rank > 2 inputs
    output = output.flatten(1, output.rank());
    data   = data.flatten(1, data.rank());

    // Add a trailing singleton dimension to the data input for slicing.
    data = data.expand({1});

    // Indices should be a vector but may contain some singleton dimensions.
    indices = indices.flatten();
    indices = indices.expand({1});
    // Assume indices are non-negative
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
  }

  snap::Tensor backward(const ScatterReduceGradOp &op,
                        const PopOpx &opx,
                        snap::Tensor &gradIn,
                        snap::Tensor &indices,
                        size_t axis,
                        snap::program::Sequence &prog,
                        const popops::SlicePlan &plan) const {
    if (op.indexBroadcasted()) {
      prepMultiSliceBroadcastedTensors(opx, gradIn, indices, axis, prog);
    } else {
      prepMultiSliceTensors(gradIn, indices, axis);
    }

    // Delegate to concrete subclasses the evaluation of the gradient.
    return calcGradient(opx, gradIn, indices, axis, prog, plan);
  }

  void prepMultiSliceBroadcastedTensors(const PopOpx &opx,
                                        snap::Tensor &data,
                                        snap::Tensor &indices,
                                        size_t axis,
                                        snap::program::Sequence &prog) const {
    // The gradient of a scatter reduction requires a gather (multiSlice).
    // Just like in the forward pass we need to support inputs with rank > 2 so:
    //   * permute dims of data and indices and output so that slice axis == 0
    //   * indices are linearized into a 1-d coordinate system
    //   * flatten the remaining dims
    data    = data.dimRoll(axis);
    indices = indices.dimRoll(axis);

    if (indices.rank() < 2) {
      data    = data.expand({1});
      indices = indices.expand({1});
    } else {
      auto numCols = indices.numElements() / indices.shape().at(0);
      indices = scatterutilx::linearizeIndices(opx, prog, indices, numCols);
      data    = data.flatten();
      data    = data.expand({1});
    }

    // Assume indices are non-negative
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
  }

  void prepMultiSliceTensors(snap::Tensor &data,
                             snap::Tensor &indices,
                             size_t axis) const {
    // Place the gather axis at the front.
    data = data.dimRoll(axis);
    data = data.flatten(1, data.rank());

    // Indices should be a vector but may contain some singleton dimensions.
    indices = indices.flatten();
    indices = indices.expand({1});
    // Assume indices are non-negative
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
  }

  virtual snap::Tensor calcGradient(const PopOpx &opx,
                                    const snap::Tensor &gradIn,
                                    const snap::Tensor &indices,
                                    size_t axis,
                                    snap::program::Sequence &prog,
                                    const popops::SlicePlan &plan) const = 0;

  virtual ~ReductionStrategy() {}
};

class SumReductionStrategy : public ReductionStrategy {
public:
  poplar::OptionFlags
  createSlicePlanOptions(const ScatterReduceOp &op) const final {
    return popart::popx::createSlicePlanOptions(
        popart::popx::SlicePlanUsedFor::UpdateAdd,
        op.getAvailableMemoryProportion());
  }

  void initReductionOutput(const PopOpx &opx,
                           const snap::Tensor &out,
                           snap::program::Sequence &prog) const final {
    popops::zero(opx.graph().getPoplarGraph(),
                 out.getPoplarTensor(),
                 prog.getPoplarSequence(),
                 opx.debugContext("zero"));
  }

  void applyReduction(const PopOpx &opx,
                      const snap::Tensor &target,
                      const snap::Tensor &update,
                      const snap::Tensor &indices,
                      snap::program::Sequence &prog,
                      const popops::SlicePlan &plan) const final {
    auto &graph = opx.graph().getPoplarGraph();
    auto scale  = graph.addConstant(
        update.elementType(), {}, 1.0f, opx.debugContext("const_1"));
    graph.setTileMapping(scale, 0);
    popops::multiUpdateAdd(graph,
                           target.getPoplarTensor(),
                           update.getPoplarTensor(),
                           indices.getPoplarTensor(),
                           scale,
                           {0},
                           {1},
                           prog.getPoplarSequence(),
                           plan,
                           poplar::OptionFlags(),
                           opx.debugContext("scatterSum"));
  }

  snap::Tensor calcGradient(const PopOpx &opx,
                            const snap::Tensor &gradIn,
                            const snap::Tensor &indices,
                            size_t axis,
                            snap::program::Sequence &prog,
                            const popops::SlicePlan &plan) const final {
    auto result = popops::multiSlice(opx.graph().getPoplarGraph(),
                                     gradIn.getPoplarTensor(),
                                     indices.getPoplarTensor(),
                                     {0},
                                     {1},
                                     prog.getPoplarSequence(),
                                     plan,
                                     poplar::OptionFlags(),
                                     opx.getDebugNameAndId("scatterSumGrad"));

    auto gradOut     = snap::Tensor{result, opx.graph()};
    auto gradOutInfo = opx.outInfo(ScatterReduceGradOp::gradOutIndex());
    return alignToAxis(gradOut, gradOutInfo.shape(), axis);
  }
};

class MaxReductionStrategy : public ReductionStrategy {
public:
  poplar::OptionFlags
  createSlicePlanOptions(const ScatterReduceOp &op) const final {
    auto options = popart::popx::createSlicePlanOptions(
        popart::popx::SlicePlanUsedFor::UpdateMax,
        op.getAvailableMemoryProportion());

    return options;
  }

  void initReductionOutput(const PopOpx &opx,
                           const snap::Tensor &out,
                           snap::program::Sequence &prog) const final {
    auto dtype = out.elementType();

    if (dtype == poplar::FLOAT || dtype == poplar::HALF) {
      initMaxTensor<float>(opx, out, prog);
      return;
    }

    if (dtype == poplar::INT) {
      initMaxTensor<int>(opx, out, prog);
      return;
    }

    if (dtype == poplar::UNSIGNED_INT) {
      initMaxTensor<unsigned int>(opx, out, prog);
      return;
    }

    throw popart::internal_error("Unsupported data type {}", dtype);
  }

  template <typename T>
  static void initMaxTensor(const PopOpx &opx,
                            const snap::Tensor &out,
                            snap::program::Sequence &prog) {
    popops::fill<T>(opx.graph().getPoplarGraph(),
                    out.getPoplarTensor(),
                    prog.getPoplarSequence(),
                    maxInitValue<T>(),
                    opx.debugContext("maxInit"));
  }

  void maskedFillOutput(const PopOpx &opx,
                        const snap::Tensor &out,
                        snap::program::Sequence &prog) const {
    poplar::Tensor init;
    poplar::Tensor zero;
    auto dtype   = out.elementType();
    auto &graph  = opx.graph().getPoplarGraph();
    auto &output = out.getPoplarTensor();
    auto &seq    = prog.getPoplarSequence();

    if (dtype == poplar::FLOAT || dtype == poplar::HALF) {
      init = graph.addConstant(
          dtype, {}, maxInitValue<float>(), opx.debugContext("maxInit"));
      zero = graph.addConstant(dtype, {}, 0.0f, opx.debugContext("zero"));
    };

    if (dtype == poplar::INT) {
      init = graph.addConstant(
          dtype, {}, maxInitValue<int>(), opx.debugContext("maxInit"));
      zero = graph.addConstant(dtype, {}, 0, opx.debugContext("zero"));
    };

    if (dtype == poplar::UNSIGNED_INT) {
      init = graph.addConstant(
          dtype, {}, maxInitValue<unsigned int>(), opx.debugContext("maxInit"));
      zero = graph.addConstant(dtype, {}, 0, opx.debugContext("zero"));
    };

    graph.setTileMapping(init, 0);
    graph.setTileMapping(zero, 0);

    auto mask =
        popops::neq(graph, output, init, seq, opx.debugContext("out!=max"));

    popops::selectInPlace(
        graph, output, zero, mask, seq, opx.debugContext("maskedFill"));
  }

  template <typename T> static T maxInitValue() {
    return std::numeric_limits<T>::has_infinity
               ? -std::numeric_limits<T>::infinity()
               : std::numeric_limits<T>::lowest();
  }

  void applyReduction(const PopOpx &opx,
                      const snap::Tensor &target,
                      const snap::Tensor &update,
                      const snap::Tensor &indices,
                      snap::program::Sequence &prog,
                      const popops::SlicePlan &plan) const final {
    popops::multiUpdateMax(opx.graph().getPoplarGraph(),
                           target.getPoplarTensor(),
                           update.getPoplarTensor(),
                           indices.getPoplarTensor(),
                           {0},
                           {1},
                           prog.getPoplarSequence(),
                           plan,
                           poplar::OptionFlags(),
                           opx.debugContext("scatter_max"));

    // TODO(T65173): make this an operator option since it can be unnecessary.
    // Replace any non-updated values with zero
    maskedFillOutput(opx, target, prog);
  }

  snap::Tensor calcGradient(const PopOpx &opx,
                            const snap::Tensor &gradIn,
                            const snap::Tensor &indices,
                            size_t axis,
                            snap::program::Sequence &prog,
                            const popops::SlicePlan &plan) const final {

    auto &graph = opx.graph().getPoplarGraph();
    auto result = popops::multiSlice(graph,
                                     gradIn.getPoplarTensor(),
                                     indices.getPoplarTensor(),
                                     {0},
                                     {1},
                                     prog.getPoplarSequence(),
                                     plan,
                                     poplar::OptionFlags(),
                                     opx.getDebugNameAndId("gatherGradIn"));

    auto gradOut     = snap::Tensor{result, opx.graph()};
    auto gradOutInfo = opx.outInfo(ScatterReduceGradOp::gradOutIndex());
    gradOut          = alignToAxis(gradOut, gradOutInfo.shape(), axis);

    auto fwdOut = opx.getInTensor(ScatterReduceGradOp::fwdOutInIndex());

    if (opx.getOp<ScatterReduceGradOp>().indexBroadcasted()) {
      fwdOut = fwdOut.dimRoll(axis);
      fwdOut = fwdOut.flatten();
      fwdOut = fwdOut.expand({1});
    } else {
      fwdOut = fwdOut.dimRoll(axis);
      fwdOut = fwdOut.flatten(1, fwdOut.rank());
    }

    auto fwdOutputs = popops::multiSlice(graph,
                                         fwdOut.getPoplarTensor(),
                                         indices.getPoplarTensor(),
                                         {0},
                                         {1},
                                         prog.getPoplarSequence(),
                                         plan,
                                         poplar::OptionFlags(),
                                         opx.getDebugNameAndId("gatherFwdOut"));

    auto outputs = snap::Tensor{fwdOutputs, opx.graph()};
    outputs      = alignToAxis(outputs, gradOutInfo.shape(), axis);

    // gradOut * (data == outputs)
    const auto &data = opx.getInTensor(ScatterReduceGradOp::dataInIndex());
    auto dtype       = gradOut.elementType();
    auto expr = pe::Mul(pe::_1, pe::Cast(pe::Equal(pe::_2, pe::_3), dtype));

    snap::popops::mapInPlace(opx.graph(),
                             expr,
                             {gradOut, data, outputs},
                             prog,
                             opx.getDebugNameAndId("gradOutMulOutputMask"));

    return gradOut;
  }
};

// TODO(T35696): When poplar supports different reduction modes we should change
// this class to target that directly rather than going through multiUpdateMax
class MinReductionStrategy : public ReductionStrategy {
public:
  poplar::OptionFlags
  createSlicePlanOptions(const ScatterReduceOp &op) const final {
    return max_strategy.createSlicePlanOptions(op);
  }

  void initReductionOutput(const PopOpx &opx,
                           const snap::Tensor &out,
                           snap::program::Sequence &prog) const final {
    max_strategy.initReductionOutput(opx, out, prog);
  }

  void applyReduction(const PopOpx &opx,
                      const snap::Tensor &target,
                      const snap::Tensor &update,
                      const snap::Tensor &indices,
                      snap::program::Sequence &prog,
                      const popops::SlicePlan &plan) const final {
    auto &graph    = opx.graph().getPoplarGraph();
    auto negUpdate = popops::neg(
        graph, update.getPoplarTensor(), prog.getPoplarSequence(), "negUpdate");

    max_strategy.applyReduction(
        opx, target, snap::Tensor(negUpdate, opx.graph()), indices, prog, plan);

    popops::negInPlace(
        graph, target.getPoplarTensor(), prog.getPoplarSequence(), "negTarget");
  }

  snap::Tensor calcGradient(const PopOpx &opx,
                            const snap::Tensor &gradIn,
                            const snap::Tensor &indices,
                            size_t axis,
                            snap::program::Sequence &prog,
                            const popops::SlicePlan &plan) const final {
    return max_strategy.calcGradient(opx, gradIn, indices, axis, prog, plan);
  }

private:
  MaxReductionStrategy max_strategy;
};

std::unique_ptr<ReductionStrategy>
createStrategy(const ScatterReduction &reduction) {
  if (reduction == ScatterReduction::Sum) {
    return std::make_unique<SumReductionStrategy>();
  }
  if (reduction == ScatterReduction::Max) {
    return std::make_unique<MaxReductionStrategy>();
  }
  if (reduction == ScatterReduction::Min) {
    return std::make_unique<MinReductionStrategy>();
  }

  throw popart::internal_error("Unsupported reduction strategy!");
}

ScatterReduceOpx::ScatterReduceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex), strategy(), plan(), axis() {
  verifyOp<ScatterReduceOp>(op, {Onnx::CustomOperators::ScatterReduce});

  auto &srop   = getOp<ScatterReduceOp>();
  strategy     = createStrategy(srop.getReduction());
  auto options = strategy->createSlicePlanOptions(srop);

  axis = static_cast<size_t>(srop.getAxis());
  nonstd::optional<size_t> plan_axis =
      srop.indexBroadcasted() ? nonstd::optional<size_t>() : axis;

  plan = createSlicePlan(graph(),
                         outInfo(srop.outIndex()),
                         inInfo(srop.indicesInIndex()),
                         options,
                         plan_axis);

  // We always want the ScatterReduce to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceOpx::grow(snap::program::Sequence &prog) const {
  const auto &srop    = getOp<ScatterReduceOp>();
  const auto &data    = getInTensor(ScatterReduceOp::dataInIndex());
  const auto &indices = getInTensor(ScatterReduceOp::indicesInIndex());

  snap::Tensor out = createDataTensor(graph(),
                                      outInfo(ScatterReduceOp::outIndex()),
                                      plan,
                                      axis,
                                      srop.indexBroadcasted(),
                                      getDebugNameAndId("scatterreduceOutput"));
  strategy->initReductionOutput(*this, out, prog);
  strategy->forward(srop, *this, out, data, indices, axis, prog, plan);
  setOutTensor(ScatterReduceOp::outIndex(), out);
}

snap::Tensor
ScatterReduceOpx::createInputTensor(InIndex index,
                                    const poplar::DebugNameAndId &dnai) const {
  if (index != ScatterReduceOp::dataInIndex() &&
      index != ScatterReduceOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  logging::debug("ScatterReduceOpx::createInputTensor index={}", index);

  auto &srop       = getOp<ScatterReduceOp>();
  auto indicesInfo = inInfo(ScatterReduceOp::indicesInIndex());

  if (index == ScatterReduceOp::indicesInIndex()) {
    return createIndicesTensor(
        graph(), indicesInfo, plan, axis, srop.indexBroadcasted(), dnai);
  }

  return createUpdateTensor(graph(),
                            inInfo(ScatterReduceOp::dataInIndex()),
                            indicesInfo,
                            plan,
                            axis,
                            srop.indexBroadcasted(),
                            dnai);
}

InputCreatorType ScatterReduceOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterReduceOp::dataInIndex() ||
      index == ScatterReduceOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return PopOpx::getInputCreatorType(index);
}

ScatterReduceGradOpx::ScatterReduceGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex), strategy(), plan(), axis() {
  verifyOp<ScatterReduceGradOp>(
      op, {Onnx::CustomGradOperators::ScatterReduceGradOp});

  auto &srop = getOp<ScatterReduceGradOp>();
  strategy   = createStrategy(srop.getReduction());

  // Gradient always requires evaluting a gather regardless of the reduction
  auto options = createSlicePlanOptions(SlicePlanUsedFor::Slice,
                                        srop.getAvailableMemoryProportion());

  axis = static_cast<size_t>(srop.getAxis());
  nonstd::optional<size_t> plan_axis =
      srop.indexBroadcasted() ? nonstd::optional<size_t>() : axis;

  plan = createSlicePlan(graph(),
                         inInfo(srop.gradInIndex()),
                         inInfo(srop.indicesInIndex()),
                         options,
                         plan_axis);

  // We always want the ScatterReduceGrad to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceGradOpx::grow(snap::program::Sequence &prog) const {
  const auto &srop = getOp<ScatterReduceGradOp>();
  auto gradIn      = getInTensor(ScatterReduceGradOp::gradInIndex());
  auto indices     = getInTensor(ScatterReduceGradOp::indicesInIndex());
  auto gradOut =
      strategy->backward(srop, *this, gradIn, indices, axis, prog, plan);

  setOutTensor(ScatterReduceGradOp::gradOutIndex(), gradOut);
}

snap::Tensor ScatterReduceGradOpx::createInputTensor(
    InIndex index,
    const poplar::DebugNameAndId &dnai) const {
  if (index != ScatterReduceGradOp::gradInIndex() &&
      index != ScatterReduceGradOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  auto &srop = getOp<ScatterReduceGradOp>();
  if (index == ScatterReduceGradOp::gradInIndex()) {
    return createDataTensor(
        graph(), inInfo(index), plan, axis, srop.indexBroadcasted(), dnai);
  }

  return createIndicesTensor(
      graph(), inInfo(index), plan, axis, srop.indexBroadcasted(), dnai);
}

InputCreatorType
ScatterReduceGradOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterReduceGradOp::gradInIndex() ||
      index == ScatterReduceGradOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }
  return PopOpx::getInputCreatorType(index);
}

namespace {
OpxCreator<ScatterReduceOpx>
    scatterReduceOpxCreator(Onnx::CustomOperators::ScatterReduce);
OpxCreator<ScatterReduceGradOpx>
    scatterReduceGradOpxCreator(Onnx::CustomGradOperators::ScatterReduceGradOp);
} // namespace

} // namespace popx
} // namespace popart
