// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <memory>
#include <set>
#include <snap/Graph.hpp>
#include <snap/Tensor.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <poplar/TensorCloneMethod.hpp>
#include <poprithms/logging/timepartitionlogger.hpp>
#include <poputil/TileMapping.hpp>
#include <popart/popx/creatorx.hpp>
#include <popart/popx/inittensor.hpp>
#include <popart/popx/irlowering.hpp>

#include "popart/debugcontext.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/opdebuginfo.hpp"
#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/linearmapper.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/popx/poptensors.hpp"
#include "popart/popx/preparedtensor.hpp"
#include "popart/popx/viewchangers.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
namespace popx {

namespace {

snap::Graph &getGraph(Tensor *t, IrLowering &irLowering) {
  auto vgid = t->getVirtualGraphIdAndTileSetUnsafe();

  if (vgid.first == unusedVGraphId) {
    return irLowering.graph();
  } else {
    return irLowering.getVirtualGraph(vgid.first, vgid.second);
  }
}

} // namespace

InitTensorBase::InitTensorBase(InitMethod method_,
                               TensorId dstId_,
                               RequireParallelWritable requireParallelWritable_,
                               double priority_)
    : method(method_), dstId(dstId_),
      requireParallelWritable(requireParallelWritable_), priority(priority_) {}

std::set<TensorId> InitTensorBase::getDependsOnIds() const {
  if (hasSrcId()) {
    return {getSrcId()};
  } else {
    return {};
  }
}

std::string InitTensorBase::str() const {
  std::stringstream ss;
  ss << "InitTensor[";

  std::string extra = extraStr();

  ss << method;

  if (!extra.empty()) {
    ss << "(" << extra << ")";
  }

  ss << ": ";
  ss << dstId << " <- {";
  auto depends = getDependsOnIds();
  ss << logging::join(depends.begin(), depends.end(), ", ");
  ss << "}]";
  return ss.str();
}

InitTensorAliasing::InitTensorAliasing(
    TensorId srcId_,
    TensorId dstId_,
    RequireParallelWritable requireParallelWritable_,
    double priority_)
    : InitTensorBase(InitMethod::Aliasing,
                     dstId_,
                     requireParallelWritable_,
                     priority_),
      srcId(srcId_) {}

bool InitTensorAliasing::initTensor(IrLowering &irLowering) const {
  if (irLowering.tensors().canAlias(getSrcId(), requireParallelWritable)) {
    logging::debug("Aliasing tensor {} to {}", srcId, getDstId());
    irLowering.tensors().insertAliased(getDstId(), getSrcId());
    return true;
  }
  return false;
}

InitTensorPostIrAliasing::InitTensorPostIrAliasing(
    TensorId srcId_,
    TensorId dstId_,
    RequireParallelWritable requireParallelWritable_,
    double priority_)
    : InitTensorBase(InitMethod::PostIrAliasing,
                     dstId_,
                     requireParallelWritable_,
                     priority_),
      srcId(srcId_) {}

bool InitTensorPostIrAliasing::initTensor(IrLowering &irLowering) const {
  if (irLowering.tensors().contains(getSrcId()) &&
      irLowering.tryInitTensorByPostIRAliasing(
          getDstId(),
          requireParallelWritable,
          irLowering.tensors().hasViewChangers(getSrcId())
              ? irLowering.tensors().getViewChangers(getSrcId())
              : ViewChangers())) {
    return true;
  }
  return false;
}

InitTensorCloning::InitTensorCloning(
    TensorId srcId_,
    TensorId dstId_,
    RequireParallelWritable requireParallelWritable_,
    double priority_)
    : InitTensorBase(InitMethod::Cloning,
                     dstId_,
                     requireParallelWritable_,
                     priority_),
      srcId(srcId_) {}

bool InitTensorCloning::initTensor(IrLowering &irLowering) const {
  if (!irLowering.tensors().contains(srcId)) {
    return false;
  }

  Tensor *t = irLowering.ir().getTensor(srcId);
  auto vgid = t->getVirtualGraphIdAndTileSetUnsafe();

  auto &dstGraph = getGraph(t, irLowering);

  logging::debug("Cloning tensor {} to {} (vgid: {} tileset: {}). Source "
                 "TensorInfo is: {}",
                 getSrcId(),
                 getDstId(),
                 vgid.first,
                 vgid.second,
                 t->info);

  auto src = irLowering.tensors().get(getSrcId());

  snap::Tensor dst;

  if (t->hasProducer()) {
    Op *producer = t->getProducer();
    dst          = dstGraph.clone(
        src,
        {poplar::DebugNameAndId(dstId,
                                producer->getDebugInfo().getId(),
                                producer->getDebugInfo().getPathName())});
  } else {
    dst = dstGraph.clone(src, {poplar::DebugNameAndId(dstId)});
  }

  if (irLowering.tensors().hasViewChangers(getSrcId())) {
    irLowering.tensors().setViewChangers(
        getDstId(), irLowering.tensors().getViewChangers(getSrcId()));
  }
  irLowering.tensors().insert(getDstId(), dst);
  return true;
}

InitTensorCreator::InitTensorCreator(
    ICreatorCandidatePtr candidate_,
    std::set<TensorId> mustExist_,
    TensorId dstId_,
    RequireParallelWritable requireParallelWritable_,
    double priority_)
    : InitTensorBase(InitMethod::Creator,
                     dstId_,
                     requireParallelWritable_,
                     priority_),
      candidate(candidate_), mustExist(mustExist_) {}

bool InitTensorCreator::initTensor(IrLowering &irLowering) const {
  for (auto tensorId : mustExist) {
    if (!irLowering.tensors().contains(tensorId)) {
      return false;
    }
  }

  const auto addOpTasksTimer =
      irLowering.ir().timePartitionLogger().scopedStopwatch(
          "Initializing Tensor Creator (Ir Lowering)");

  logging::devicex::debug(
      "Creating poplar::Tensor {}, with layout allocated by {}",
      getDstId(),
      candidate->str());

  Tensor *tensor = irLowering.ir().getTensor(getDstId());

  auto inputAndView = candidate->createInput(
      {poplar::DebugNameAndId(getDstId() + "_tmp",
                              tensor->getDebugInfo().getId(),
                              tensor->getDebugInfo().getPathName())});

  // Try if an existing Poplar tensor can be reused
  if (irLowering.tryInitTensorByPostIRAliasing(
          getDstId(), requireParallelWritable, inputAndView.second)) {
    return true;
  }

  if (!inputAndView.second.empty()) {
    // Underlying poplar::Tensor does not match IR expectations, supply
    // view-changing transformation
    irLowering.tensors().setViewChangers(getDstId(), inputAndView.second);
  }

  logging::devicex::trace("Cloning poplar::Tensor {}.", getDstId());

  // The clone makes sure to only keep the necessary parts of the unwound
  // tensor alive, and contiguate it,
  // reducing IPU memory liveness and fragmentation (see T18661)
  auto input = irLowering.graph().clone(
      inputAndView.first,
      {poplar::DebugNameAndId(getDstId(),
                              tensor->getDebugInfo().getId(),
                              tensor->getDebugInfo().getPathName())});

  irLowering.tensors().insert(getDstId(), input);
  irLowering.addEfficientlyCreatedInputTensors(getDstId());
  return true;
}

std::string InitTensorCreator::extraStr() const { return candidate->str(); }

std::set<TensorId> InitTensorCreator::getDependsOnIds() const {
  return mustExist;
}

InitTensorLinear::InitTensorLinear(
    TensorId dstId_,
    RequireParallelWritable requireParallelWritable,
    double priority_)
    : InitTensorBase(InitMethod::Linear,
                     dstId_,
                     requireParallelWritable,
                     priority_) {}

bool InitTensorLinear::initTensor(IrLowering &irLowering) const {
  // Try if an existing Poplar tensor can be reused
  if (irLowering.tryInitTensorByPostIRAliasing(
          getDstId(), requireParallelWritable, ViewChangers())) {
    return true;
  }
  Tensor *tensor = irLowering.ir().getTensor(getDstId());

  auto vgid = tensor->getVirtualGraphIdAndTileSetUnsafe();

  logging::devicex::debug(
      "Creating poplar::Tensor '{}' linearly on vgid: {}, tileset: {}. No "
      "operator specific allocator found",
      tensor->id,
      vgid.first,
      vgid.second);

  auto &dstGraph = getGraph(tensor, irLowering);

  auto dataType = tensor->info.dataType();

  if (irLowering.ir().getSessionOptions().enableSupportedDataTypeCasting) {
    dataType = getCompatibleDataType(dataType);
  }

  auto newTensor = dstGraph.addVariable(
      popType(dataType),
      tensor->info.shape_szt(),
      {poplar::DebugNameAndId(tensor->str(),
                              tensor->getDebugInfo().getId(),
                              tensor->getDebugInfo().getPathName())});
  irLowering.getLinearMapper().mapTensor(dstGraph, newTensor);

  irLowering.tensors().insert(getDstId(), newTensor);
  irLowering.addLinearlyCreatedInputTensors(tensor->id);
  return true;
}

InitTensorRTS::InitTensorRTS(TensorId srcId_,
                             TensorId dstId_,
                             RequireParallelWritable requireParallelWritable,
                             double priority_)
    : InitTensorBase(InitMethod::ReplicatedTensorSharding,
                     dstId_,
                     requireParallelWritable,
                     priority_),
      srcId(srcId_) {}

bool InitTensorRTS::initTensor(IrLowering &irLowering) const {
  if (!irLowering.tensors().contains(srcId)) {
    return false;
  }

  // Try if an existing Poplar tensor can be reused
  if (irLowering.tryInitTensorByPostIRAliasing(
          getDstId(), requireParallelWritable, ViewChangers())) {
    return true;
  }

  Tensor *srcTensor = irLowering.ir().getTensor(getSrcId());
  Tensor *dstTensor = irLowering.ir().getTensor(getDstId());

  auto vgid = dstTensor->getVirtualGraphIdAndTileSetUnsafe();

  logging::devicex::debug("Creating poplar::Tensor '{}' with replicated tensor "
                          "sharding on vgid: {}, tileset: {}.",
                          dstTensor->id,
                          vgid.first,
                          vgid.second);

  auto &srcGraph = getGraph(srcTensor, irLowering);
  auto &dstGraph = getGraph(dstTensor, irLowering);

  auto newTensor = poputil::cloneToGraph(
      srcGraph.getPoplarGraph(),
      dstGraph.getPoplarGraph(),
      irLowering.tensors().get(srcId).getPoplarTensor(),
      {poplar::DebugNameAndId(dstTensor->str(),
                              dstTensor->getDebugInfo().getId(),
                              dstTensor->getDebugInfo().getPathName())},
      poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);

  if (irLowering.tensors().hasViewChangers(getSrcId())) {
    irLowering.tensors().setViewChangers(
        getDstId(), irLowering.tensors().getViewChangers(getSrcId()));
  }
  irLowering.tensors().insert(getDstId(), {newTensor, dstGraph});
  irLowering.addEfficientlyCreatedInputTensors(dstTensor->id);
  return true;
}

std::ostream &operator<<(std::ostream &os, const InitMethod &method) {
  switch (method) {
  case InitMethod::None: {
    os << "None";
    break;
  }
  case InitMethod::Aliasing: {
    os << "Aliasing";
    break;
  }
  case InitMethod::PostIrAliasing: {
    os << "PostIrAliasing";
    break;
  }
  case InitMethod::Cloning: {
    os << "Cloning";
    break;
  }
  case InitMethod::Creator: {
    os << "Creator";
    break;
  }
  case InitMethod::Linear: {
    os << "Linear";
    break;
  }
  case InitMethod::ReplicatedTensorSharding: {
    os << "ReplicatedTensorSharding";
    break;
  }
  }
  return os;
}

} // namespace popx
} // namespace popart
