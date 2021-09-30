// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPX_RNG_RNG_STATE_HELPER_HPP
#define GUARD_NEURALNET_POPX_RNG_RNG_STATE_HELPER_HPP

#include <poplar/CSRFunctions.hpp>
#include <poplar/RandomSeed.hpp>

#include <poprand/RandomGen.hpp>

#include <popops/ElementWise.hpp>

#include <popx/rng/rngstatelowering.hpp>

namespace popart {
namespace popx {

RngStateLowering::RngStateLowering(const IrLowering &irLowering_,
                                   snap::Graph &graph_)
    : irLowering{irLowering_}, graph{graph_}, differingSeedsRngStateTensor{},
      identicalSeedsRngStateTensor{} {

  // Create the PRNG state tensor tensors.
  auto workersPerTile =
      graph.get().getPoplarGraph().getTarget().getNumWorkerContexts();
  auto numTiles = graph.get().getPoplarGraph().getTarget().getNumTiles();

  // Create tensor to hold the inacive RNG state.
  differingSeedsRngStateTensor =
      snap::Tensor{graph.get().getPoplarGraph().addVariable(
                       poplar::UNSIGNED_INT,
                       {numTiles, workersPerTile, 4},
                       {"differingSeedsRngStateTensor"}),
                   graph.get()};
  // Layout tensor carefully to avoid exchanges.
  setTensorLayout(differingSeedsRngStateTensor);

  // Create tensor to hold the inacive RNG state.
  identicalSeedsRngStateTensor =
      snap::Tensor{graph.get().getPoplarGraph().addVariable(
                       poplar::UNSIGNED_INT,
                       {numTiles, workersPerTile, 4},
                       {"differingSeedsRngStateTensor"}),
                   graph.get()};
  // Layout tensor carefully to avoid exchanges.
  setTensorLayout(identicalSeedsRngStateTensor);
}

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
}

void RngStateLowering::lowerSetRngState(snap::program::Sequence &seq,
                                        PopOpx *opx) {

  Op *op = opx->op_p;

  if (op->hasStochasticRoundingMethod()) {
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

  if (op->hasStochasticRoundingMethod()) {
    if (op->getStochasticRoundingMethod() ==
        StochasticRoundingMethod::DifferingSeeds) {
      lowerGetHwSeeds(seq,
                      identicalSeedsRngStateTensor,
                      opx->debugContext("lowerGetRngState/DifferingSeeds"));
    } else if (op->getStochasticRoundingMethod() ==
               StochasticRoundingMethod::IdenticalSeeds) {
      lowerGetHwSeeds(seq,
                      identicalSeedsRngStateTensor,
                      opx->debugContext("lowerSetRngState/IdenticalSeeds"));
    }
  }
}

void RngStateLowering::setTensorLayout(snap::Tensor &tensor) {

  auto numTiles = graph.get().getPoplarGraph().getTarget().getNumTiles();
  if (tensor.rank() >= 1 && tensor.shape()[0] == numTiles) {

    for (auto tile = 0U; tile != numTiles; ++tile) {
      auto slice = tensor.slice({tile, tile + 1}, 0);
      graph.get().getPoplarGraph().setTileMapping(slice.getPoplarTensor(),
                                                  tile);
    }

  } else {
    throw internal_error("[RngStateLowering] Expected tensor with first "
                         "dimension of {} (got tensor shape {})",
                         numTiles,
                         tensor.shape());
  }
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
  seq.add(poplar::program::Copy(
      tmp.getPoplarTensor(), rngState.getPoplarTensor(), false, dbgCtx));
}

} // namespace popx
} // namespace popart

#endif
