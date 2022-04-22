// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPX_RNG_RNG_STATE_HELPER_HPP
#define GUARD_NEURALNET_POPX_RNG_RNG_STATE_HELPER_HPP

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <snap/Graph.hpp>
#include <snap/Program.hpp>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/RandomSeed.hpp>
#include <poplar/ReplicatedStreamMode.hpp>
#include <poplar/Target.hpp>
#include <poplar/Type.hpp>
#include <popops/ElementWise.hpp>
#include <poprand/RandomGen.hpp>
#include <popx/rng/rngstatelowering.hpp>
#include <popart/op/getrandomseed.hpp>
#include <popart/popx/pritask.hpp>

#include "popart/devicemanager.hpp"
#include "popart/ir.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/popx/irlowering.hpp"
#include "popart/popx/linearmapper.hpp"
#include "popart/popx/popopx.hpp"
#include "popart/popx/popprograms.hpp"
#include "popart/popx/poptensors.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/taskid.hpp"

namespace popart {
namespace popx {

RngStateLowering::RngStateLowering(IrLowering &irLowering_, snap::Graph &graph_)
    : irLowering{irLowering_}, graph{graph_}, differingSeedsRngStateTensor{},
      identicalSeedsRngStateTensor{} {

  // Create the PRNG state tensor tensors.
  // Create tensor to hold the inacive RNG state.
  differingSeedsRngStateTensor =
      createRNGStateTensor(graph.get(), "differingSeedsRngStateTensor");

  // Create tensor to hold the inacive RNG state.
  identicalSeedsRngStateTensor =
      createRNGStateTensor(graph.get(), "identicalSeedsRngStateTensor");
}

RngStateLowering::~RngStateLowering() = default;

void RngStateLowering::lowerInitRngStatesFromSeed(
    snap::program::Sequence &seq,
    const snap::Tensor &seed,
    const poplar::DebugContext &dbgCtx) {

  // The call to `setSeed` below is what triggers deriving the RNG state from
  // `seed`. Note that at this point the value of `seed` should be identical
  // across replicas (this is a precondition of this function) and hence the
  // derived RNG state will also be identical across replicas.
  poprand::setSeed(graph.get().getPoplarGraph(),
                   seed.getPoplarTensor(),
                   0,
                   seq.getPoplarSequence(),
                   dbgCtx);

  // Copy the replica-identical RNG state into the tensor we use to hold the
  // inactive RNG state. We will call `poplar::setHwSeeds` with this tensor
  // later before Ops with stochastic rounding method
  // `StochasticRoundingMethod::IdenticalSeeds` are ran.
  lowerGetHwSeeds(seq, identicalSeedsRngStateTensor, dbgCtx);

  // Now update the RNG state in a replica-differing way (without affecting
  // the seed). We get an offset value that's different for each replica, add
  // it to the seed and call `setSeed` to derive a replica-differing RNG state
  // from this value.
  auto offset = graph.get().getPoplarGraph().addReplicationIndexConstant();
  graph.get().getPoplarGraph().setTileMapping(offset, 0);
  auto replicaDifferentValue = popops::add(graph.get().getPoplarGraph(),
                                           seed.getPoplarTensor(),
                                           offset,
                                           seq.getPoplarSequence(),
                                           dbgCtx);
  poprand::setSeed(graph.get().getPoplarGraph(),
                   replicaDifferentValue,
                   0,
                   seq.getPoplarSequence(),
                   dbgCtx);

  // Copy the replica-differing RNG state into the tensor we use to hold the
  // inactive RNG state. We will call `poplar::setHwSeeds` with this tensor
  // later before Ops with stochastic rounding method
  // `StochasticRoundingMethod::DifferingSeeds` are ran.
  lowerGetHwSeeds(seq, differingSeedsRngStateTensor, dbgCtx);

  // Set initial RNG state to IdenticalSeeds.
  lowerSetHwSeeds(seq, identicalSeedsRngStateTensor, dbgCtx);
}

void RngStateLowering::lowerSetRngState(snap::program::Sequence &seq,
                                        PopOpx *opx) {

  Op *op = opx->op_p;

  if (op->hasStochasticRoundingMethod() &&
      // TODO(T48752): Remove _enableRngStateManagement.
      op->getIr().getSessionOptions()._enableRngStateManagement) {
    if (op->getStochasticRoundingMethod() ==
        StochasticRoundingMethod::DifferingSeeds) {
      lowerSetHwSeeds(seq,
                      differingSeedsRngStateTensor,
                      opx->debugContext("lowerSetRngState/DifferingSeeds"));
    } else if (op->getStochasticRoundingMethod() ==
               StochasticRoundingMethod::IdenticalSeeds) {
      lowerSetHwSeeds(seq,
                      identicalSeedsRngStateTensor,
                      opx->debugContext("lowerSetRngState/IdenticalSeeds"));
    }
  }
}

void RngStateLowering::lowerGetRngState(snap::program::Sequence &seq,
                                        PopOpx *opx) {

  Op *op = opx->op_p;

  if (op->hasStochasticRoundingMethod() &&
      // TODO(T48752): Remove _enableRngStateManagement.
      op->getIr().getSessionOptions()._enableRngStateManagement) {
    if (op->getStochasticRoundingMethod() ==
        StochasticRoundingMethod::DifferingSeeds) {
      lowerGetHwSeeds(seq,
                      differingSeedsRngStateTensor,
                      opx->debugContext("lowerGetRngState/DifferingSeeds"));
    } else if (op->getStochasticRoundingMethod() ==
               StochasticRoundingMethod::IdenticalSeeds) {
      lowerGetHwSeeds(seq,
                      identicalSeedsRngStateTensor,
                      opx->debugContext("lowerGetRngState/IdenticalSeeds"));
    }
  }
}

snap::Tensor RngStateLowering::createRNGStateTensor(snap::Graph &graph,
                                                    const std::string &name) {
  auto &target                            = graph.getTarget();
  std::vector<size_t> rngStateTensorShape = getRngStateTensorShape(graph);
  // Create tensor with specific mapping to avoid exchanges.
  unsigned minElementsPerTile =
      target.getNumWorkerContexts() * rngStateSizePerWorker;
  return graph.addLinearlyMappedVariable(poplar::UNSIGNED_INT,
                                         rngStateTensorShape,
                                         minElementsPerTile,
                                         minElementsPerTile,
                                         {name});
}

void RngStateLowering::lowerSetHwSeeds(
    snap::program::Sequence &seq,
    snap::Tensor &rngState,
    const poplar::DebugContext &dbgCtx) const {
  poplar::setHwSeeds(graph.get().getPoplarGraph(),
                     rngState.getPoplarTensor(),
                     seq.getPoplarSequence(),
                     dbgCtx);
}

void RngStateLowering::lowerGetHwSeeds(
    snap::program::Sequence &seq,
    snap::Tensor &rngState,
    const poplar::DebugContext &dbgCtx) const {
  auto tmp = snap::Tensor{poplar::getHwSeeds(graph.get().getPoplarGraph(),
                                             seq.getPoplarSequence(),
                                             dbgCtx),
                          graph.get()};
  seq.add(snap::program::Copy(tmp, rngState, false, dbgCtx));
}

PriTask RngStateLowering::initRngStateTensor() {
  // Set up combinedRngStateTensor
  auto initRngStateTensorTask = [this]() {
    SequenceMap seqs(graph.get());

    std::vector<size_t> combinedRngStateTensorShape =
        getCombinedRngStateTensorShape(graph.get());
    combinedRngStateTensor =
        graph.get().addVariable(poplar::UNSIGNED_INT,
                                combinedRngStateTensorShape,
                                {"combinedRngStateTensor"});
    irLowering.get().getLinearMapper().mapTensor(graph.get(),
                                                 combinedRngStateTensor);
    return SequenceMap(graph.get());
  };
  return {+1e6, initRngStateTensorTaskId, {}, initRngStateTensorTask};
}

PriTask RngStateLowering::randomSeedToHost() {
  auto streamedSeedId = GetRandomSeedOp::getStreamedSeedTensorId();

  auto randomSeedToHostTask = [this, streamedSeedId]() {
    // Copy the value of the 'streamed seed' back to the host.
    auto &streamedSeed = irLowering.get().tensors().get(streamedSeedId);

    auto streamSeedToHost = graph.get().addDeviceToHostFIFO(
        "d2h_randomSeed", poplar::UNSIGNED_INT, 2);

    logging::devicex::debug("Initializing random seed d2h.");

    SequenceMap seqs(graph.get());
    auto &seq =
        seqs.getSequence(&irLowering.get().progs().randomSeedToHostFragment());

    seq.add(snap::program::Copy(
        streamedSeed, streamSeedToHost, false, {"randomSeedToHost"}));
    return seqs;
  };

  std::vector<PriTaskDependency> deps;
  deps.push_back(irLowering.get().taskWhichCreates(streamedSeedId));

  return {0, randomSeedToHostTaskId, deps, randomSeedToHostTask};
}

PriTask RngStateLowering::rngStateFromHost() {
  auto rngStateFromHostTask = [this]() {
    const size_t combinedRngTensorSize =
        getCombinedRngStateTensorSize(graph.get());

    auto streamRngFromHost = graph.get().addHostToDeviceFIFO(
        "h2d_rngStateTensor",
        poplar::UNSIGNED_INT,
        combinedRngTensorSize,
        poplar::ReplicatedStreamMode::REPLICATE);

    logging::devicex::debug("Initializing RNG h2d with RNG size equal to {}.",
                            combinedRngTensorSize);

    SequenceMap seqs(graph.get());
    auto &seq =
        seqs.getSequence(&irLowering.get().progs().rngStateFromHostFragment());

    // Stream newRngState to combinedRngStateTensor
    seq.add(snap::program::Copy(streamRngFromHost,
                                combinedRngStateTensor,
                                false,
                                {"copyStreamRngStateTensor"}));
    // Copy first half of combinedRngStateTensor to identicalSeedsRngStateTensor
    seq.add(snap::program::Copy(
        combinedRngStateTensor[0],
        identicalSeedsRngStateTensor,
        false,
        {"copyRngStateTensorToIdenticalSeedsRngStateTensor"}));
    // Copy second half of combinedRngStateTensor to
    // differingSeedsRngStateTensor
    seq.add(snap::program::Copy(
        combinedRngStateTensor[1],
        differingSeedsRngStateTensor,
        false,
        {"copyRngStateTensorToDifferingSeedsRngStateTensor"}));
    return seqs;
  };
  return {0,
          rngStateFromHostTaskId,
          {{initRngStateTensorTaskId, DependencyType::Tensor}},
          rngStateFromHostTask};
}

PriTask RngStateLowering::rngStateToHost() {
  auto rngStateToHostTask = [this]() {
    const size_t combinedRngTensorSize =
        getCombinedRngStateTensorSize(graph.get());

    auto streamRngToHost = graph.get().addDeviceToHostFIFO(
        "d2h_rngStateTensor", poplar::UNSIGNED_INT, combinedRngTensorSize);

    logging::devicex::debug("Initializing RNG d2h with RNG size equal to {}.",
                            combinedRngTensorSize);

    SequenceMap seqs(graph.get());
    auto &seq =
        seqs.getSequence(&irLowering.get().progs().rngStateToHostFragment());

    // Update combinedRngStateTensor with the new values of
    // identicalSeedsRngStateTensor and differingSeedsRngStateTensor
    seq.add(snap::program::Copy(
        snap::concat(identicalSeedsRngStateTensor.expand({0}),
                     differingSeedsRngStateTensor.expand({0})),
        combinedRngStateTensor,
        false,
        "seedsToRngStateTensor"));
    // Stream combinedRngStateTensor to host
    seq.add(snap::program::Copy(
        combinedRngStateTensor, streamRngToHost, false, {"rngStateToHost"}));
    return seqs;
  };

  return {0,
          rngStateToHostTaskId,
          {{initRngStateTensorTaskId, DependencyType::Tensor}},
          rngStateToHostTask};
}

std::vector<size_t>
RngStateLowering::getCombinedRngStateTensorShape(const snap::Graph &graph) {
  std::vector<size_t> rngTensorShape = getRngStateTensorShape(graph);
  rngTensorShape.insert(rngTensorShape.begin(), numRngStateTensors);
  return rngTensorShape;
}

size_t
RngStateLowering::getCombinedRngStateTensorSize(const snap::Graph &graph) {
  const std::vector<size_t> tensorShape = getCombinedRngStateTensorShape(graph);
  return std::accumulate(tensorShape.begin(),
                         tensorShape.end(),
                         size_t(1),
                         std::multiplies<size_t>());
}

std::vector<size_t> RngStateLowering::getCombinedRngStateTensorShape(
    const popart::DeviceInfo &deviceInfo,
    const unsigned replicationFactor) {
  std::vector<size_t> rngTensorShape =
      getRngStateTensorShape(deviceInfo, replicationFactor);
  rngTensorShape.insert(rngTensorShape.begin(), numRngStateTensors);
  return rngTensorShape;
}

size_t RngStateLowering::getCombinedRngStateTensorSize(
    const popart::DeviceInfo &deviceInfo,
    const unsigned replicationFactor) {
  const std::vector<size_t> tensorShape =
      getCombinedRngStateTensorShape(deviceInfo, replicationFactor);
  return std::accumulate(tensorShape.begin(),
                         tensorShape.end(),
                         size_t(1),
                         std::multiplies<size_t>());
}

std::vector<size_t>
RngStateLowering::getRngStateTensorShape(const snap::Graph &graph) {
  const size_t numTiles       = graph.getTarget().getNumTiles();
  const size_t workersPerTile = graph.getTarget().getNumWorkerContexts();
  return std::vector<size_t>{numTiles, workersPerTile, rngStateSizePerWorker};
}

std::vector<size_t>
RngStateLowering::getRngStateTensorShape(const popart::DeviceInfo &deviceInfo,
                                         const unsigned replicationFactor) {
  const size_t numTiles =
      deviceInfo.getTarget().getNumTiles() / replicationFactor;
  const size_t workersPerTile = deviceInfo.getTarget().getNumWorkerContexts();
  return std::vector<size_t>{numTiles, workersPerTile, rngStateSizePerWorker};
}

const TaskId RngStateLowering::initRngStateTensorTaskId =
    TaskId(TaskId::Type::InitRngStateTensorTask);
const TaskId RngStateLowering::randomSeedToHostTaskId =
    TaskId(TaskId::Type::RandomSeedToHostTask);
const TaskId RngStateLowering::rngStateFromHostTaskId =
    TaskId(TaskId::Type::RngStateFromHostTask);
const TaskId RngStateLowering::rngStateToHostTaskId =
    TaskId(TaskId::Type::RngStateToHostTask);

} // namespace popx
} // namespace popart

#endif
