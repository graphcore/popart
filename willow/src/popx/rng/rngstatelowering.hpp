// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_POPX_RNG_RNG_STATE_LOWERING_HPP
#define GUARD_NEURALNET_POPX_RNG_RNG_STATE_LOWERING_HPP

#include <popart/op.hpp>
#include <popart/popx/irlowering.hpp>

#include <poplar/Program.hpp>

#include <snap/Graph.hpp>

namespace popart {
namespace popx {

/**
 * This class is responsible for lowering RNG/SR related Poplar calls to Poplar
 * sequences. With 'RNG state' we mean the state you can load/store with
 * Poplar's `getHwSeed/setHwSeed` functions and with SR seed we mean the
 * enabledness of stochastic rounding.
 *
 * When stochastic rounding is enabled the assumption we make is that
 * `stochasticRoundingMethod` attribute is set for all Ops. This class offers
 * methods to set
 *
 * NOTE: We have deliberately chosen to avoid adding this functionality in to
 * IrLowering to avoid adding too much into one class.
 **/
class RngStateLowering {
public:
  /**
   * Initialise a RngStateLowering object.
   */
  RngStateLowering(const IrLowering &irLowering, snap::Graph &graph);
  virtual ~RngStateLowering();

  /**
   * Lower the code required to initialise the RNG state tensors from a new
   * seed.
   *
   * ASSUMPTION: The value of the seed tensor passed inis expected to be
   * identical across replicas.
   *
   * In pseudo-code:
   * ```
   * def lowerInitRngStatesFromSeed(userSeed):
   *   poplar::setSeed(userSeed)
   *   identicalSeedsRngStateTensor = poplar::getHwSeeds()
   *   userSeed += replicationIndexConstant
   *   differingSeedsRngStateTensor = poplar::getHwSeeds()
   * ```
   *
   * @param seq The Poplar sequence to lower into.
   * @param seed The Poplar tensor (the user's seed).
   * @param dbgCtx The debug context to use.
   */
  virtual void lowerInitRngStatesFromSeed(snap::program::Sequence &seq,
                                          const snap::Tensor &seed,
                                          const poplar::DebugContext &dbgCtx);

  /**
   * Lower the Poplar logic required to set the RNG state before the growing
   * of an Op.
   *
   * In pseudo-code:
   * ```
   * def lowerSetRngState(op):
   *   if op.settings.stochasticRoundingMethod == IdenticalSeeds:
   *     poplar::setHwSeeds(identicalSeedsRngStateTensor)
   *   if op.settings.stochasticRoundingMethod == DifferingSeeds:
   *     poplar::setHwSeeds(differingSeedsRngStateTensor)
   * ```
   *
   * @param seq The poplar sequence to load the RNG state change into.
   * @param op The Op prior for which we're making the change.
   */
  virtual void lowerSetRngState(snap::program::Sequence &seq, PopOpx *opx);

  /**
   * Lower the Poplar logic required to get the RNG state after the growing
   * of an Op.
   *
   * In pseudo-code:
   * ```
   * def lowerGetRngState(op):
   *   if op.settings.stochasticRoundingMethod == IdenticalSeeds:
   *     identicalSeedsRngStateTensor = poplar::getHwSeeds()
   *   if op.settings.stochasticRoundingMethod == DifferingSeeds:
   *     differingSeedsRngStateTensor = oplar::setHwSeeds()
   * ```
   *
   * @param seq The poplar sequence to load the RNG state change into.
   * @param op The Op prior for which we're making the change.
   */
  virtual void lowerGetRngState(snap::program::Sequence &seq, PopOpx *opx);

private:
  // Helper function to ensure our tensors have a layout that prevents
  // unnecessary exchanges.
  void setTensorLayout(snap::Tensor &tensor);

  // Helper function for calling `setHwSeeds` with `rngState`.
  virtual void lowerSetHwSeeds(snap::program::Sequence &seq,
                               snap::Tensor &rngState,
                               const poplar::DebugContext &dbgCtx) const;

  // Helper function for calling `getHwSeeds` with `rngState`.
  virtual void lowerGetHwSeeds(snap::program::Sequence &seq,
                               snap::Tensor &rngState,
                               const poplar::DebugContext &dbgCtx) const;

  // Reference to IR.
  std::reference_wrapper<const IrLowering> irLowering;
  // Reference to the snap Graph.
  std::reference_wrapper<snap::Graph> graph;

  // We maintain two RNG states, one that is identical between replicas and one
  // that is not identical.
  snap::Tensor differingSeedsRngStateTensor;
  snap::Tensor identicalSeedsRngStateTensor;
};

} // namespace popx
} // namespace popart

#endif
