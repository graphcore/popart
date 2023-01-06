// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPX_RNG_RNG_STATE_HELPER_HPP_
#define GUARD_NEURALNET_POPX_RNG_RNG_STATE_HELPER_HPP_

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/RandomSeed.hpp>
#include <poplar/ReplicatedStreamMode.hpp>
#include <poplar/Target.hpp>
#include <poplar/Type.hpp>
#include <popops/ElementWise.hpp>
#include <poprand/RandomGen.hpp>
#include <poputil/TileMapping.hpp>
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
#include "popart/popx/opx.hpp"
#include "popart/popx/popprograms.hpp"
#include "popart/popx/poptensors.hpp"
#include "popart/sessionoptions.hpp"
#include "popart/taskid.hpp"

namespace popart {
namespace popx {

RngStateLowering::RngStateLowering(IrLowering &irLowering_,
                                   poplar::Graph &graph_)
    : irLowering{irLowering_}, graph{graph_}, differingSeedsRngStateTensor{},
      identicalSeedsRngStateTensor{} {

  // Create the PRNG state tensor tensors.
  // Create tensor to hold the inactive RNG state.
  differingSeedsRngStateTensor =
      createRNGStateTensor(graph.get(), "differingSeedsRngStateTensor");

  // Create tensor to hold the inactive RNG state.
  identicalSeedsRngStateTensor =
      createRNGStateTensor(graph.get(), "identicalSeedsRngStateTensor");
}

RngStateLowering::~RngStateLowering() = default;

void RngStateLowering::lowerInitRngStatesFromSeed(
    poplar::program::Sequence &seq,
    const poplar::Tensor &seed,
    const poplar::DebugContext &dbgCtx) {

  // The call to `setSeed` below is what triggers deriving the RNG state from
  // `seed`. Note that at this point the value of `seed` should be identical
  // across replicas (this is a precondition of this function) and hence the
  // derived RNG state will also be identical across replicas.
  poprand::setSeed(graph.get(), seed, 0, seq, dbgCtx);

  // Copy the replica-identical RNG state into the tensor we use to hold the
  // inactive RNG state. We will call `poplar::setHwSeeds` with this tensor
  // later before Ops with stochastic rounding method
  // `StochasticRoundingMethod::IdenticalSeeds` are ran.
  lowerGetHwSeeds(seq, identicalSeedsRngStateTensor, dbgCtx);

  // Now update the RNG state in a replica-differing way (without affecting
  // the seed). We get an offset value that's different for each replica, add
  // it to the seed and call `setSeed` to derive a replica-differing RNG state
  // from this value.
  auto offset = graph.get().addReplicationIndexConstant();
  graph.get().setTileMapping(offset, 0);
  auto replicaDifferentValue =
      popops::add(graph.get(), seed, offset, seq, dbgCtx);
  poprand::setSeed(graph.get(), replicaDifferentValue, 0, seq, dbgCtx);

  // Copy the replica-differing RNG state into the tensor we use to hold the
  // inactive RNG state. We will call `poplar::setHwSeeds` with this tensor
  // later before Ops with stochastic rounding method
  // `StochasticRoundingMethod::DifferingSeeds` are ran.
  lowerGetHwSeeds(seq, differingSeedsRngStateTensor, dbgCtx);

  // Set initial RNG state to IdenticalSeeds.
  lowerSetHwSeeds(seq, identicalSeedsRngStateTensor, dbgCtx);
}

void RngStateLowering::lowerSetRngState(poplar::program::Sequence &seq,
                                        Opx *opx) {

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

void RngStateLowering::lowerGetRngState(poplar::program::Sequence &seq,
                                        Opx *opx) {

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

poplar::Tensor RngStateLowering::createRNGStateTensor(poplar::Graph &graph,
                                                      const std::string &name) {
  auto &target                            = graph.getTarget();
  std::vector<size_t> rngStateTensorShape = getRngStateTensorShape(graph);
  // Create tensor with specific mapping to avoid exchanges.
  unsigned minElementsPerTile =
      target.getNumWorkerContexts() * rngStateSizePerWorker;
  auto result =
      graph.addVariable(poplar::UNSIGNED_INT, rngStateTensorShape, {name});
  poputil::mapTensorLinearly(
      graph, result, minElementsPerTile, minElementsPerTile);
  return result;
}

void RngStateLowering::lowerSetHwSeeds(
    poplar::program::Sequence &seq,
    poplar::Tensor &rngState,
    const poplar::DebugContext &dbgCtx) const {
  poplar::setHwSeeds(graph.get(), rngState, seq, dbgCtx);
}

void RngStateLowering::lowerGetHwSeeds(
    poplar::program::Sequence &seq,
    poplar::Tensor &rngState,
    const poplar::DebugContext &dbgCtx) const {
  auto tmp = poplar::getHwSeeds(graph.get(), seq, dbgCtx);
  seq.add(poplar::program::Copy(tmp, rngState, false, dbgCtx));
}

PriTask RngStateLowering::initRngStateTensor() {
  // Set up combinedRngStateTensor
  auto initRngStateTensorTask = [this]() {
    SequenceMap seqs;

    std::vector<size_t> combinedRngStateTensorShape =
        getCombinedRngStateTensorShape(graph.get());
    combinedRngStateTensor =
        graph.get().addVariable(poplar::UNSIGNED_INT,
                                combinedRngStateTensorShape,
                                {"combinedRngStateTensor"});
    irLowering.get().getLinearMapper().mapTensor(graph.get(),
                                                 combinedRngStateTensor);
    return SequenceMap();
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

    SequenceMap seqs;
    auto &seq =
        seqs.getSequence(&irLowering.get().progs().randomSeedToHostFragment());

    seq.add(poplar::program::Copy(
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

    SequenceMap seqs;
    auto &seq =
        seqs.getSequence(&irLowering.get().progs().rngStateFromHostFragment());

    // Stream newRngState to combinedRngStateTensor
    seq.add(poplar::program::Copy(streamRngFromHost,
                                  combinedRngStateTensor,
                                  false,
                                  {"copyStreamRngStateTensor"}));
    // Copy first half of combinedRngStateTensor to identicalSeedsRngStateTensor
    seq.add(poplar::program::Copy(
        combinedRngStateTensor[0],
        identicalSeedsRngStateTensor,
        false,
        {"copyRngStateTensorToIdenticalSeedsRngStateTensor"}));
    // Copy second half of combinedRngStateTensor to
    // differingSeedsRngStateTensor
    seq.add(poplar::program::Copy(
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

    SequenceMap seqs;
    auto &seq =
        seqs.getSequence(&irLowering.get().progs().rngStateToHostFragment());

    // Update combinedRngStateTensor with the new values of
    // identicalSeedsRngStateTensor and differingSeedsRngStateTensor
    seq.add(poplar::program::Copy(
        poplar::concat(identicalSeedsRngStateTensor.expand({0}),
                       differingSeedsRngStateTensor.expand({0})),
        combinedRngStateTensor,
        false,
        "seedsToRngStateTensor"));
    // Stream combinedRngStateTensor to host
    seq.add(poplar::program::Copy(
        combinedRngStateTensor, streamRngToHost, false, {"rngStateToHost"}));
    return seqs;
  };

  return {0,
          rngStateToHostTaskId,
          {{initRngStateTensorTaskId, DependencyType::Tensor}},
          rngStateToHostTask};
}

std::vector<size_t>
RngStateLowering::getCombinedRngStateTensorShape(const poplar::Graph &graph) {
  std::vector<size_t> rngTensorShape = getRngStateTensorShape(graph);
  rngTensorShape.insert(rngTensorShape.begin(), numRngStateTensors);
  return rngTensorShape;
}

size_t
RngStateLowering::getCombinedRngStateTensorSize(const poplar::Graph &graph) {
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
RngStateLowering::getRngStateTensorShape(const poplar::Graph &graph) {
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
