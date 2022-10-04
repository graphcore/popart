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
  createForwardPlanOptions(const ScatterReduceOp &op) const = 0;

  virtual poplar::OptionFlags
  createBackwardPlanOptions(const ScatterReduceGradOp &op) const = 0;

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

  std::vector<snap::Tensor> backward(const ScatterReduceGradOp &op,
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

    // Delegate to concrete subclasses the evaluation of the gradient(s).
    return calcGradient(op, opx, gradIn, indices, axis, prog, plan);
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

  virtual std::vector<snap::Tensor>
  calcGradient(const ScatterReduceGradOp &op,
               const PopOpx &opx,
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
  createForwardPlanOptions(const ScatterReduceOp &op) const final {
    return createSlicePlanOptions(SlicePlanUsedFor::UpdateAdd,
                                  op.getAvailableMemoryProportion());
  }

  poplar::OptionFlags
  createBackwardPlanOptions(const ScatterReduceGradOp &op) const final {
    return createSlicePlanOptions(SlicePlanUsedFor::Slice,
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

  std::vector<snap::Tensor>
  calcGradient(const ScatterReduceGradOp &op,
               const PopOpx &opx,
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

    auto gradData = snap::Tensor{result, opx.graph()};
    auto outInfo  = opx.outInfo(ScatterReduceGradOp::gradDataOutIndex());
    gradData      = alignToAxis(gradData, outInfo.shape(), axis);

    if (op.hasInitialValues()) {
      auto gradInitials = opx.cloneNcopy(
          prog, opx.getInTensor(ScatterReduceGradOp::gradInIndex()));
      return {gradData, gradInitials};
    }

    return {gradData};
  }
};

class NoneReductionStrategy : public ReductionStrategy {
public:
  poplar::OptionFlags
  createForwardPlanOptions(const ScatterReduceOp &op) const final {
    return createSlicePlanOptions(SlicePlanUsedFor::Update,
                                  op.getAvailableMemoryProportion());
  }

  poplar::OptionFlags
  createBackwardPlanOptions(const ScatterReduceGradOp &op) const final {
    if (op.hasInitialValues()) {
      // none-reduction with initial values needs both a scatter and a gather
      // for each gradient output.
      return createSlicePlanOptions(SlicePlanUsedFor::CombinedSliceUpdate,
                                    op.getAvailableMemoryProportion());
    }

    return createSlicePlanOptions(SlicePlanUsedFor::Slice,
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
    popops::multiUpdate(opx.graph().getPoplarGraph(),
                        target.getPoplarTensor(),
                        update.getPoplarTensor(),
                        indices.getPoplarTensor(),
                        {0},
                        {1},
                        prog.getPoplarSequence(),
                        plan,
                        poplar::OptionFlags(),
                        opx.debugContext("scatter"));
  }

  std::vector<snap::Tensor>
  calcGradient(const ScatterReduceGradOp &op,
               const PopOpx &opx,
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
                                     opx.getDebugNameAndId("scatterNoneGrad"));

    auto gradData        = snap::Tensor{result, opx.graph()};
    auto gradDataOutInfo = opx.outInfo(ScatterReduceGradOp::gradDataOutIndex());
    gradData             = alignToAxis(gradData, gradDataOutInfo.shape(), axis);

    if (!op.hasInitialValues()) {
      return {gradData};
    }

    // Scatter of zeros into the gradIn
    auto dataInfo    = opx.inInfo(ScatterReduceGradOp::dataInIndex());
    auto indicesInfo = opx.inInfo(ScatterReduceGradOp::indicesInIndex());

    auto numSlices  = static_cast<size_t>(dataInfo.nelms());
    auto outputSize = 1UL;

    if (!op.indexBroadcasted()) {
      numSlices  = dataInfo.shape_szt().at(axis);
      outputSize = dataInfo.nelms() / numSlices;
    }

    auto numLookups = static_cast<size_t>(indicesInfo.nelms());

    auto zeros =
        popops::createSliceTensor(opx.graph().getPoplarGraph(),
                                  gradIn.elementType(),
                                  {numSlices, outputSize},
                                  {0},
                                  {1},
                                  numLookups,
                                  plan,
                                  poplar::OptionFlags(),
                                  opx.getDebugNameAndId("zerosUpdate"));

    popops::fill(opx.graph().getPoplarGraph(),
                 zeros,
                 prog.getPoplarSequence(),
                 0.0f,
                 opx.debugContext("zerosUpdateFill"));

    auto t =
        popops::createSliceableTensor(opx.graph().getPoplarGraph(),
                                      gradIn.elementType(),
                                      {gradIn.dim(0), gradIn.dim(1)},
                                      {0},
                                      {1},
                                      plan,
                                      poplar::OptionFlags(),
                                      opx.getDebugNameAndId("zerosUpdate"));

    auto gradInitials = snap::Tensor{t, opx.graph()};
    prog.getPoplarSequence().add(snap::program::Copy(
        gradIn, gradInitials, false, opx.debugContext("copyToScatter")));

    popops::multiUpdate(opx.graph().getPoplarGraph(),
                        gradInitials.getPoplarTensor(),
                        zeros,
                        indices.getPoplarTensor(),
                        {0},
                        {1},
                        prog.getPoplarSequence(),
                        plan,
                        poplar::OptionFlags(),
                        opx.getDebugNameAndId("scatterInitialValuesGrad"));

    auto info = opx.outInfo(ScatterReduceGradOp::gradInitialValuesOutIndex());
    gradInitials = alignToAxis(gradInitials, info.shape(), axis);
    return {gradData, gradInitials};
  }
};

class MaxReductionStrategy : public ReductionStrategy {
public:
  poplar::OptionFlags
  createForwardPlanOptions(const ScatterReduceOp &op) const final {
    return createSlicePlanOptions(SlicePlanUsedFor::UpdateMax,
                                  op.getAvailableMemoryProportion());
  }

  poplar::OptionFlags
  createBackwardPlanOptions(const ScatterReduceGradOp &op) const final {
    return createSlicePlanOptions(SlicePlanUsedFor::Slice,
                                  op.getAvailableMemoryProportion());
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
                        const snap::Tensor &src,
                        const snap::Tensor &indices,
                        snap::program::Sequence &prog,
                        const popops::SlicePlan &plan) const {
    auto &graph = opx.graph().getPoplarGraph();
    auto &seq   = prog.getPoplarSequence();

    auto mask = opx.cloneNcopy(prog, out);
    popops::zero(graph, mask.getPoplarTensor(), seq, "zeros");

    auto ones = opx.cloneNcopy(prog, src);
    popops::fill(graph, ones.getPoplarTensor(), seq, 1, "ones");

    popops::multiUpdateMax(graph,
                           mask.getPoplarTensor(),
                           ones.getPoplarTensor(),
                           indices.getPoplarTensor(),
                           {0},
                           {1},
                           seq,
                           plan,
                           poplar::OptionFlags(),
                           opx.debugContext("scatter_mask"));

    auto expr = pe::Select(pe::_1, pe::_2, pe::Cast(pe::_2, poplar::BOOL));
    snap::popops::mapInPlace(
        opx.graph(), expr, {out, mask}, prog, opx.debugContext("maskedFill"));
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

    if (!opx.hasInput(ScatterReduceOp::initialValuesInIndex())) {
      // TODO(T65173): make this an operator option since it can be unnecessary.
      // Replace any non-updated values with zero
      maskedFillOutput(opx, target, update, indices, prog, plan);
    }
  }

  std::vector<snap::Tensor>
  calcGradient(const ScatterReduceGradOp &op,
               const PopOpx &opx,
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

    auto gradData     = snap::Tensor{result, opx.graph()};
    auto gradDataInfo = opx.outInfo(ScatterReduceGradOp::gradDataOutIndex());
    gradData          = alignToAxis(gradData, gradDataInfo.shape(), axis);

    auto fwdOut = opx.getInTensor(ScatterReduceGradOp::fwdOutInIndex());

    if (op.indexBroadcasted()) {
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
    outputs      = alignToAxis(outputs, gradDataInfo.shape(), axis);

    // gradDataOut * (data == outputs)
    const auto &data = opx.getInTensor(ScatterReduceGradOp::dataInIndex());
    auto dtype       = gradData.elementType();
    auto expr = pe::Mul(pe::_1, pe::Cast(pe::Equal(pe::_2, pe::_3), dtype));

    snap::popops::mapInPlace(opx.graph(),
                             expr,
                             {gradData, data, outputs},
                             prog,
                             opx.getDebugNameAndId("gradOutMulOutputMask"));

    if (!op.hasInitialValues()) {
      return {gradData};
    }

    // gradIn * (fwdOut == initialValues)
    auto grad = opx.getInTensor(ScatterReduceGradOp::gradInIndex());
    auto fwd  = opx.getInTensor(ScatterReduceGradOp::fwdOutInIndex());
    auto initialValues =
        opx.getInTensor(ScatterReduceGradOp::initialValuesInIndex());

    auto gradInitials =
        snap::popops::map(opx.graph(),
                          expr,
                          {grad, fwd, initialValues},
                          prog,
                          opx.getDebugNameAndId("gradInMulInitialValuesMask"));

    return {gradData, gradInitials};
  }
};

// TODO(T35696): When poplar supports different reduction modes we should change
// this class to target that directly rather than going through multiUpdateMax
class MinReductionStrategy : public ReductionStrategy {
public:
  poplar::OptionFlags
  createForwardPlanOptions(const ScatterReduceOp &op) const final {
    return max_strategy.createForwardPlanOptions(op);
  }

  poplar::OptionFlags
  createBackwardPlanOptions(const ScatterReduceGradOp &op) const final {
    return max_strategy.createBackwardPlanOptions(op);
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
    auto &graph = opx.graph().getPoplarGraph();

    if (opx.hasInput(ScatterReduceOp::initialValuesInIndex())) {
      popops::negInPlace(graph,
                         target.getPoplarTensor(),
                         prog.getPoplarSequence(),
                         "negTarget");
    }

    auto negUpdate = popops::neg(
        graph, update.getPoplarTensor(), prog.getPoplarSequence(), "negUpdate");

    max_strategy.applyReduction(
        opx, target, snap::Tensor(negUpdate, opx.graph()), indices, prog, plan);

    popops::negInPlace(
        graph, target.getPoplarTensor(), prog.getPoplarSequence(), "negTarget");
  }

  std::vector<snap::Tensor>
  calcGradient(const ScatterReduceGradOp &op,
               const PopOpx &opx,
               const snap::Tensor &gradIn,
               const snap::Tensor &indices,
               size_t axis,
               snap::program::Sequence &prog,
               const popops::SlicePlan &plan) const final {
    return max_strategy.calcGradient(
        op, opx, gradIn, indices, axis, prog, plan);
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
  if (reduction == ScatterReduction::None) {
    return std::make_unique<NoneReductionStrategy>();
  }

  throw popart::internal_error("Unsupported reduction strategy!");
}

ScatterReduceOpx::ScatterReduceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex), strategy(), plan(), axis() {
  verifyOp<ScatterReduceOp>(op, {Onnx::CustomOperators::ScatterReduce});

  auto &srop   = getOp<ScatterReduceOp>();
  strategy     = createStrategy(srop.getReduction());
  auto options = strategy->createForwardPlanOptions(srop);

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

  if (srop.hasInput(ScatterReduceOp::initialValuesInIndex())) {
    const auto &t = getInTensor(ScatterReduceOp::initialValuesInIndex());
    prog.getPoplarSequence().add(snap::program::Copy(
        t, out, false, debugContext("copyToScatterReduce")));
  } else {
    strategy->initReductionOutput(*this, out, prog);
  }
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

  auto &srop   = getOp<ScatterReduceGradOp>();
  strategy     = createStrategy(srop.getReduction());
  auto options = strategy->createBackwardPlanOptions(srop);

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

  if (gradOut.size() != srop.outTensorCount()) {
    throw error("ScatterReduceGradOpx must calculate at least one gradient "
                " tensor and no more than two.");
  }

  setOutTensor(ScatterReduceGradOp::gradDataOutIndex(), gradOut[0]);

  if (srop.hasInitialValues()) {
    setOutTensor(ScatterReduceGradOp::gradInitialValuesOutIndex(), gradOut[1]);
  }
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
