// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_EXCHANGE_HPP
#define GUARD_NEURALNET_EXCHANGE_HPP

#include <popart/op.hpp>
#include <popart/op/elementwise.hpp>

namespace popart {

/**
 * Enum type to specify an exchange strategy
 *
 * ==============================================================
 * JustInTime:
 * .- outer loop -------------.
 * |.- inner loop -----------.|
 * || load - compute - store ||
 * |'------------------------'|
 * '--------------------------'
 *
 * ==============================================================
 * OverlapInnerLoop:
 * - Boxes denote subgraphs / subgraph Ops / loops
 * - Inputs/outputs are loop carried in order
 *
 * .- outer loop ----------------------------------------.
 * |                  .- inner loop -.                   |
 * | load - compute - | - store      |                   |
 * |           load - | - compute -- | - store           |
 * |                  |   load ----- | - compute - store |
 * |                  '--------------'                   |
 * '-----------------------------------------------------'
 *          ^^^^^^^       ^^^^^^^        ^^^^^^^
 *          overlap       overlap        overlap
 *
 * ==============================================================
 * OverlapLoops
 * - Boxes denote subgraphs / subgraph Ops / loops
 * - Numbers on boxes are matching subgraph/loop inputs and outputs
 * - Overlap indicators indicate compute & load/store pairs overlapping in time
 *
 *                load
 *                  |
 *               compute   load            load         < overlap
 *                  |        |               |
 *                  1        2               |
 *              .-- inner loop --.           |
 *              |   |        |   |           |
 *              | store  compute |           |          < overlap
 *              | load       |   |           |          < overlap
 *              |   |        |   |           |
 *              '----------------'           |
 *                  2        1      load compute        < overlap
 *                  |        |        |      |
 *                  1        2        3      4
 * .- outer loop -----------------------------------.
 * |                |        |        |      |      |
 * |             compute   store      |    store    |   < overlap
 * |                \                /              |
 * |                 1              2               |
 * |                .-- inner loop --.              |
 * |                |   |        |   |              |
 * |                | store  compute |              |   < overlap
 * |                | load       |   |              |   < overlap
 * |                |   |        |   |              |
 * |                '----------------'              |
 * |                    2        1                  |
 * |                    |        |                  |
 * |    load        compute      |       load       |   < overlap
 * |      |             |        |         |        |
 * '------------------------------------------------'
 *        3             4        2         1
 *        |             |        |         |
 *    compute           |      store       |            < overlap
 *        |              \                /
 *        |               1              2
 *        |              .-- inner loop --.
 *        |              |   |        |   |
 *        |              | store  compute |             < overlap
 *        |              | load       |   |             < overlap
 *        |              |   |        |   |
 *        |              '----------------'
 *        |                  2        1
 *        |                  |        |
 *     store             compute    store               < overlap
 *                           |
 *                         store
 *
 * ==============================================================
 * OverlapStep:
 * Not supported yet
 */
enum class ExchangeStrategy {
  /// Copy tensor when required
  JustInTime = 0,
  /// Preload values in previous inner loop iteration for the next iteration
  OverlapInnerLoop = 1,
  /// Preload values in the previous loop iteration for the next iteration
  /// (implies OverlapInnerLoop)
  OverlapLoops = 2,
  /// Preload values in the previous host training step for next step
  /// (implies OverlapLoops) - not supported yet
  OverlapStep = 3,
  /// Number of values
  N = 4
};

/**
 * Enum type to specify an exchange direction
 */
enum class ExchangeDirection {
  /// Copy into IPU on-chip memory
  Load = 0,
  /// Copy out of IPU on-chip memory
  Store = 1,
  /// Number of values
  N = 2
};

std::ostream &operator<<(std::ostream &, const ExchangeStrategy &);

/**
 * Class describing an external exchanges from IPUs
 */
class ExchangeDescriptor {
public:
  /**
   * Create an ExchangeDescriptor for a host exchange
   * \param direction \p Load (from host) or \p Store (to host)
   * \param id Host stream tensor ID
   * \param vgid Virtual graph for the exchange
   * \param tileSet Tile set for the exchange
   * \param numInputs Number of tensor inputs expected
   * \param numOutputs Number of tensor outputs expected
   */
  ExchangeDescriptor(ExchangeDirection direction,
                     TensorId id,
                     OptionalVGraphId vgid,
                     TileSet tileSet,
                     int numInputs,
                     int numOutputs);

  /**
   * Create an ExchangeDescriptor for a remote exchange
   * \param direction \p Load (from host) or \p Store (to host)
   * \param id Remote buffer id
   * \param vgid Virtual graph for the exchange
   * \param tileSet Tile set for the exchange
   * \param numInputs Number of tensor inputs expected
   * \param numOutputs Number of tensor outputs expected
   */
  ExchangeDescriptor(ExchangeDirection direction,
                     RemoteBufferId id,
                     OptionalVGraphId vgid,
                     TileSet tileSet,
                     int numInputs,
                     int numOutputs);

  const ExchangeDirection &getDirection() const { return direction; }

  bool isRemoteExchange() const { return hostStreamTensorId.empty(); }

  bool isHostExchange() const { return !hostStreamTensorId.empty(); }

  const RemoteBufferId &getRemoteBufferId() const { return remoteBufferId; }
  void setRemoteBufferId(RemoteBufferId id) { remoteBufferId = id; }

  const TensorId &getHostStreamTensorId() const { return hostStreamTensorId; }

  /**
   * Get an identifier representing which resource (landing pad tensor) this
   * exchange will be using
   * \return Resource identifier
   */
  const std::string getResourceId() const;

  OptionalVGraphId getVGraphID() const { return vgid; }
  TileSet getTileSet() const { return tileSet; }

  int getNumInputs() const { return numInputs; }
  int getNumOutputs() const { return numOutputs; }

private:
  /// To IPU (load) or from IPU (store)
  ExchangeDirection direction;

  /// Only set for remote exchanges
  RemoteBufferId remoteBufferId;

  /// Only set for host exchanges
  TensorId hostStreamTensorId;

  // Placement attributes
  /// Virtual graph for the exchange
  OptionalVGraphId vgid;
  /// Tile set for the exchange
  TileSet tileSet;

  /// Number of inputs required
  int numInputs;

  /// Number of outputs required
  int numOutputs;
};

class ExchangeBaseOp : public Op {
public:
  ExchangeBaseOp(const OperatorIdentifier &_opid, const Op::Settings &settings)
      : Op(_opid, settings) {}

  virtual int getNumExchanges() const { return 1; }

  /**
   * Return the exchange descriptor at index
   * A \p MultiExchangeOp can contain multiple descriptors, while
   * \p RemoteLoad/Store and \p HostLoad/Store contain one each.
   * \param index Index of the exchange descriptor to return.
   * \return \p ExchangeDescriptor for the exchange.
   */
  virtual ExchangeDescriptor getExchangeDescriptor(int index) const = 0;

  float getSubgraphValue() const final { return getHighSubgraphValue(); }
  bool isOutlineable() const final { return true; }

private:
};

} // namespace popart

#endif
