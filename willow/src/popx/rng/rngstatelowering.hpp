// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_POPX_RNG_RNGSTATELOWERING_HPP_
#define POPART_WILLOW_SRC_POPX_RNG_RNGSTATELOWERING_HPP_

#include <functional>
#include <snap/Tensor.hpp>
#include <stddef.h>
#include <string>
#include <vector>
#include <poplar/Target.hpp>

#include "popart/popx/debugcontextx.hpp"
#include "popart/popx/pritask.hpp"

namespace snap {
class Graph;

namespace program {
class Sequence;
} // namespace program
} // namespace snap

namespace popart {
class TaskId;
class DeviceInfo;

namespace popx {
class IrLowering;
class PopOpx;

/**
 * This class is responsible for lowering RNG/SR related Poplar calls to Poplar
 * sequences. With 'RNG state' we mean the state you can load/store with
 * Poplar's `getHwSeed/setHwSeed` functions. When using stochastic rounding, we
 * maintain two RNG states, one that is shared by all replicas, and one that is
 * not. This class implements methods to lower the logic for managing two
 * distinct RNG states.
 *
 * When stochastic rounding is enabled the assumption we make is that
 * `stochasticRoundingMethod` attribute is set for all Ops.
 *
 * NOTE: We have deliberately chosen to avoid adding this functionality in to
 * IrLowering to avoid adding too much into one class.
 **/
class RngStateLowering {
public:
  /**
   * Initialise a RngStateLowering object.
   */
  RngStateLowering(IrLowering &irLowering, snap::Graph &graph);
  virtual ~RngStateLowering();

  /**
   * Lower the code required to initialise the RNG state tensors from a new
   * seed.
   *
   * ASSUMPTION: The value of the seed tensor passed in is identical across
   * replicas.
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

  /**
   * @return Task to set up combinedRngStateTensor
   */
  PriTask initRngStateTensor();

  /**
   * @return Task to set up lowering random seed to host stream copy:
   */
  PriTask randomSeedToHost();

  /**
   * @return Task to set up lowering of rng state tensors from host to device
   * The lowered program is equivalent to the following pseudocode:
   * ```
   * def rngStateToHostLowered():
   *   combinedRngStateTensor = [
   *     identicalSeedsRngStateTensor,
   *     differingSeedsRngStateTensor
   *   ]
   *   rngState = combinedRngStateTensor
   * ```
   */
  PriTask rngStateFromHost();

  /**
   * @return Task to set up lowering of rng state tensors from device to host
   * The lowered program is equivalent to the following pseudocode:
   * ```
   * def rngStateFromHostLowered():
   *   combinedRngStateTensor = newRngState
   *   identicalSeedsRngStateTensor = combinedRngStateTensor[0]
   *   differingSeedsRngStateTensor = combinedRngStateTensor[1]
   * ```
   */
  PriTask rngStateToHost();

  /**
   * @param graph This object represents a graph program to be
   * executed on the IPU. We need poplar::Target from it. Note that
   * poplar::Target from graph contains target for single replicas.
   * @return Shape for combined rng state tensor. It looks as follows:
   * - First dim: The number of rng state tensors. Currently we have
   *   two rng state tensors: identicalSeedsRngStateTensor,
   *   differingSeedsRngStateTensor
   * - Second dim: The number of tiles used by single replica.
   * - Third dim: The number of workers existing on a single tile.
   * - Fourth dim: Size of single RNG state in bytes
   */
  static std::vector<size_t>
  getCombinedRngStateTensorShape(const snap::Graph &graph);

  /**
   * @param graph This object represents a graph program to be
   * executed on the IPU. We need poplar::Target from it. Note that
   * poplar::Target from graph contains target for single replicas.
   * @return Flattened size of combined rng state tensor.
   */
  static size_t getCombinedRngStateTensorSize(const snap::Graph &graph);

  /**
   * @param deviceInfo Information about the device which session
   * uses. We need poplar::Target from it. Note that poplar::Target
   * from deviceInfo contains target for all replicas.
   * @param replicationFactor is the number of replicas and the
   * value is defined by SessionOptions::replicatedGraphCount.
   * @return Shape for combined rng state tensor. It looks as follows:
   * - First dim: The number of rng state tensors. Currently we have
   *   two rng state tensors: identicalSeedsRngStateTensor,
   *   differingSeedsRngStateTensor
   * - Second dim: The number of tiles used by single replica.
   * - Third dim: The number of workers existing on a single tile.
   * - Fourth dim: Size of single RNG state in bytes
   */
  static std::vector<size_t>
  getCombinedRngStateTensorShape(const popart::DeviceInfo &deviceInfo,
                                 const unsigned replicationFactor);

  /**
   * @param deviceInfo Information about the device which session
   * uses. We need poplar::Target from it. Note that poplar::Target
   * from deviceInfo contains target for all replicas.
   * @param replicationFactor is the number of replicas and the
   * value is defined by SessionOptions::replicatedGraphCount.
   * @return Flattened size of combined rng state tensor.
   */
  static size_t
  getCombinedRngStateTensorSize(const popart::DeviceInfo &deviceInfo,
                                const unsigned replicationFactor);

  // 2 tensors: identicalSeedsRngStateTensor and differingSeedsRngStateTensor
  static const unsigned numRngStateTensors = 2;
  // Size of single RNG state in bytes
  static const unsigned rngStateSizePerWorker = 4;

protected:
  // Helper function to create a tensor to hold the inacive RNG state.
  static snap::Tensor createRNGStateTensor(snap::Graph &graph,
                                           const std::string &name);

private:
  // Helper function for calling `setHwSeeds` with `rngState`.
  virtual void lowerSetHwSeeds(snap::program::Sequence &seq,
                               snap::Tensor &rngState,
                               const poplar::DebugContext &dbgCtx) const;

  // Helper function for calling `getHwSeeds` with `rngState`.
  virtual void lowerGetHwSeeds(snap::program::Sequence &seq,
                               snap::Tensor &rngState,
                               const poplar::DebugContext &dbgCtx) const;

  /**
   * @param graph This object represents a graph program to be
   * executed on the IPU. We need poplar::Target from it. Note that
   * poplar::Target from graph contains target for single replicas.
   * @return Shape for combined rng state tensor. It looks as follows:
   * - First dim: The number of tiles used by single replica.
   * - Second dim: The number of workers existing on a single tile.
   * - Third dim: Size of single RNG state in bytes
   */
  static std::vector<size_t> getRngStateTensorShape(const snap::Graph &graph);

  /**
   * @param deviceInfo Information about the device which session
   * uses. We need poplar::Target from it. Note that poplar::Target
   * from deviceInfo contains target for all replicas.
   * @param replicationFactor is the number of replicas and the
   * value is defined by SessionOptions::replicatedGraphCount.
   * @return Shape for combined rng state tensor. It looks as follows:
   * - First dim: The number of tiles used by single replica.
   * - Second dim: The number of workers existing on a single tile.
   * - Third dim: Size of single RNG state in bytes
   */
  static std::vector<size_t>
  getRngStateTensorShape(const popart::DeviceInfo &deviceInfo,
                         const unsigned replicationFactor);

  // Reference to IR.
  std::reference_wrapper<IrLowering> irLowering;
  // Reference to the snap Graph.
  std::reference_wrapper<snap::Graph> graph;

  // We maintain two RNG states, one that is identical between replicas and one
  // that is not identical.
  snap::Tensor differingSeedsRngStateTensor;
  snap::Tensor identicalSeedsRngStateTensor;

  // Tensor used to upload / download the value [identicalSeedsRngStateTensor,
  // differingSeedsRngStateTensor].
  snap::Tensor combinedRngStateTensor;

  // TaskIds for the different tasks
  static const TaskId initRngStateTensorTaskId;
  static const TaskId randomSeedToHostTaskId;
  static const TaskId rngStateFromHostTaskId;
  static const TaskId rngStateToHostTaskId;
};

} // namespace popx
} // namespace popart

#endif // POPART_WILLOW_SRC_POPX_RNG_RNGSTATELOWERING_HPP_
