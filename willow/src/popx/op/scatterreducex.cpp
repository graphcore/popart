// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "popart/popx/debugcontextx.hpp"
#include <cstddef>
#include <limits>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <snap/Tensor.hpp>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Type.hpp>
#include <popops/DynamicSlice.hpp>
#include <popops/Fill.hpp>
#include <popart/error.hpp>
#include <popart/op/scatterreduce.hpp>
#include <popart/popx/op/scatterreducex.hpp>
#include <popart/popx/op/scatterutilx.hpp>
#include <popart/popx/op/sliceplanx.hpp>
#include <popart/popx/opxmanager.hpp>

#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {
class Op;

namespace popx {
class Devicex;

ScatterReduceOpx::ScatterReduceOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex), plan(), axis() {
  verifyOp<ScatterReduceOp>(op, {Onnx::CustomOperators::ScatterReduce});

  auto &srop   = getOp<ScatterReduceOp>();
  axis         = static_cast<size_t>(srop.getAxis());
  auto options = createSlicePlanOptions(SlicePlanUsedFor::UpdateAdd,
                                        srop.getAvailableMemoryProportion());

  if (srop.indexBroadcasted()) {
    plan = createSlicePlan(graph(),
                           outInfo(srop.outIndex()),
                           inInfo(srop.indicesInIndex()),
                           options);
  } else {
    plan = createSlicePlan(graph(),
                           outInfo(srop.outIndex()),
                           inInfo(srop.indicesInIndex()),
                           options,
                           srop.getAxis());
  }

  // We always want the ScatterReduce to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceOpx::grow(snap::program::Sequence &prog) const {
  auto &srop   = getOp<ScatterReduceOp>();
  auto data    = getInTensor(ScatterReduceOp::dataInIndex());
  auto indices = getInTensor(ScatterReduceOp::indicesInIndex());

  snap::Tensor out;
  if (srop.indexBroadcasted()) {
    out = createDataTensor(graph(),
                           outInfo(ScatterReduceOp::outIndex()),
                           plan,
                           axis,
                           getDebugNameAndId("scatterreduceOutput"));
  } else {
    out = createDataTensor(graph(),
                           outInfo(ScatterReduceOp::outIndex()),
                           plan,
                           getDebugNameAndId("scatterreduceOutput"));
  }

  popops::fill(graph().getPoplarGraph(),
               out.getPoplarTensor(),
               prog.getPoplarSequence(),
               0.0f,
               debugContext("scatterreduceFill"));

  auto scale = graph().getPoplarGraph().addConstant(
      data.elementType(), {}, 1.0f, debugContext("constOne"));
  graph().getPoplarGraph().setTileMapping(scale, 0);

  // The popops::multiUpdateAdd op is roughly:
  //   for i indices:
  //    out[indices[i]] += data[i]
  // but the output must be 2d. To support inputs with rank > 2 we do:
  //   * permute dims of data and indices and output so that slice axis == 0
  //   * indices are linearized into a 1-d coordinate system
  //   * flatten the remaining dims
  //
  // The above results in each slice being of size 1 so we lower 2-d cases
  // where index is not yet broadcasted differently, by passing the original
  // input tensors to the popops::multiUpdateAdd op directly.
  if (srop.indexBroadcasted()) {
    auto target = out.dimRoll(axis);
    data        = data.dimRoll(axis);
    indices     = indices.dimRoll(axis);

    if (indices.rank() < 2) {
      // popops::multiUpdateAdd requires 2-d inputs
      target  = target.expand({1});
      indices = indices.expand({1});
      data    = data.expand({1, 1});
    } else {
      target           = target.flatten();
      target           = target.expand({1});
      data             = data.flatten(1, data.rank());
      auto numDataCols = static_cast<int>(data.dim(1));
      indices =
          scatterutilx::linearizeIndices(*this, prog, indices, numDataCols);
      data = data.flatten();
      data = data.expand({1, 1});
    }

    // Assume indices are non-negative
    indices = indices.reinterpret(poplar::UNSIGNED_INT);

    popops::multiUpdateAdd(graph().getPoplarGraph(),
                           target.getPoplarTensor(),
                           data.getPoplarTensor(),
                           indices.getPoplarTensor(),
                           scale,
                           {0},
                           {1},
                           prog.getPoplarSequence(),
                           plan,
                           poplar::OptionFlags(),
                           debugContext("scatterAdd"));
  } else {
    auto data_shape = inInfo(ScatterReduceOp::dataInIndex()).shape_szt();
    data_shape.insert(data_shape.begin() + 1, 1);
    data    = data.reshape(data_shape);
    indices = indices.reinterpret(poplar::UNSIGNED_INT);
    popops::multiUpdateAdd(graph().getPoplarGraph(),
                           out.getPoplarTensor(),
                           data.getPoplarTensor(),
                           indices.getPoplarTensor(),
                           scale,
                           {0},
                           {1},
                           prog.getPoplarSequence(),
                           plan,
                           poplar::OptionFlags(),
                           debugContext("scatterAdd"));
  }

  setOutTensor(ScatterReduceOp::outIndex(), out);
}

snap::Tensor
ScatterReduceOpx::createInputTensor(InIndex index,
                                    const poplar::DebugNameAndId &dnai) const {
  if (index != ScatterReduceOp::dataInIndex() &&
      index != ScatterReduceOp::indicesInIndex()) {
    throw error("ScatterReduceOpx::createInput : Invalid index = {}", index);
  }

  auto &srop       = getOp<ScatterReduceOp>();
  auto indicesInfo = inInfo(ScatterReduceOp::indicesInIndex());

  if (index == ScatterReduceOp::indicesInIndex()) {
    if (srop.indexBroadcasted()) {
      return createIndicesTensor(graph(), indicesInfo, plan, axis, dnai);
    }
    return createIndicesTensor(graph(), indicesInfo, plan, dnai);
  }

  auto dataInfo = inInfo(ScatterReduceOp::dataInIndex());

  if (srop.indexBroadcasted()) {
    return createUpdateTensor(graph(), dataInfo, indicesInfo, plan, axis, dnai);
  }
  return createUpdateTensor(graph(), dataInfo, plan, dnai);
}

InputCreatorType ScatterReduceOpx::getInputCreatorType(InIndex index) const {
  if (index == ScatterReduceOp::dataInIndex() ||
      index == ScatterReduceOp::indicesInIndex()) {
    return InputCreatorType::CanCreate;
  }

  return PopOpx::getInputCreatorType(index);
}

ScatterReduceGradOpx::ScatterReduceGradOpx(Op *op, Devicex *devicex)
    : PopOpx(op, devicex) {
  verifyOp<ScatterReduceGradOp>(
      op, {Onnx::CustomGradOperators::ScatterReduceGradOp});

  auto &srop   = getOp<ScatterReduceGradOp>();
  axis         = static_cast<size_t>(srop.getAxis());
  auto options = createSlicePlanOptions(SlicePlanUsedFor::Slice,
                                        srop.getAvailableMemoryProportion());

  if (srop.indexBroadcasted()) {
    plan = createSlicePlan(graph(),
                           inInfo(srop.gradInIndex()),
                           inInfo(srop.indicesInIndex()),
                           options);
  } else {
    plan = createSlicePlan(graph(),
                           inInfo(srop.gradInIndex()),
                           inInfo(srop.indicesInIndex()),
                           options,
                           srop.getAxis());
  }

  // We always want the ScatterReduceGrad to layout its inputs
  inputCreatorPriority = std::numeric_limits<double>::max();
}

void ScatterReduceGradOpx::grow(snap::program::Sequence &prog) const {
  auto &srop       = getOp<ScatterReduceGradOp>();
  auto gradIn      = getInTensor(ScatterReduceGradOp::gradInIndex());
  auto indices     = getInTensor(ScatterReduceGradOp::indicesInIndex());
  auto gradOutInfo = outInfo(ScatterReduceGradOp::gradOutIndex());

  snap::Tensor gradOut;
  if (srop.indexBroadcasted()) {
    gradOut = scatterutilx::growScatterUpdateGrad(
        *this,
        prog,
        graph(),
        gradIn,
        indices,
        gradOutInfo.shape(),
        axis,
        plan,
        getDebugNameAndId("scatterAddGrad"));
  } else {
    gradOut = scatterutilx::growScatterUpdateGrad(
        prog,
        graph(),
        gradIn,
        indices,
        gradOutInfo,
        plan,
        getDebugNameAndId("scatterAddGrad"));
  }

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
    if (srop.indexBroadcasted()) {
      return createDataTensor(graph(), inInfo(index), plan, axis, dnai);
    }
    return createDataTensor(graph(), inInfo(index), plan, dnai);
  }

  if (srop.indexBroadcasted()) {
    return createIndicesTensor(graph(), inInfo(index), plan, axis, dnai);
  }
  return createIndicesTensor(graph(), inInfo(index), plan, dnai);
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
