// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
/// \file dataflow.hpp
#ifndef GUARD_NEURALNET_DATAFLOW_HPP
#define GUARD_NEURALNET_DATAFLOW_HPP

#include <set>
#include <string>
#include <vector>
#include <popart/names.hpp>
#include <popart/op/exchange/exchange.hpp>
#include <popart/replicatedstreammode.hpp>

namespace popart {

// Forward declaration.
struct SessionOptions;

/**
 * Class that defines the identifiers for the return type of the anchor tensors.
 *
 * An anchor tensor is a tensor that the user wants returned after a call to
 * Session::run(). Each call to Session::run() results in `batchesPerStep x
 * accumulationFactor x replicationFactor` of anchor tensors being computed. The
 * samples associated with each computation is called a micro batch. The
 * dimensions are user-specified with the following parameters:
 *
 *  * `batchesPerStep` is number of batches per step and the value is obtained
 *        from the DataFlow object.
 *  * `accumulationFactor` is the gradient accumulation factor and the value is
 *        defined by SessionOptions::accumulationFactor.
 *  * `replicationFactor` is the number of replicas and the value is defined by
 *        SessionOptions::replicatedGraphCount.
 *
 * This enum type describes the strategy with which the micro batch values for
 * anchor tensors (or their summaries) are written or to the IStepIO instance
 * passed to Session::run.
 *
 * \sa AnchorReturnType.
 *
 * **NOTE**: Anchors are essentially what TensorFlow calls "fetches".
 */
enum class AnchorReturnTypeId {
  /**
   * Only return the tensor value for the last micro batch of the Session::run
   * call for each replica.
   *
   * The buffer shape required for this anchor in IStepIO is
   * [`replicationFactor`, `<anchorTensorShape>`] (with dimensions of size 1
   * removed).
   */
  Final = 0,
  /**
   * Return the tensor value for every *N*-th global batch for each replica
   * and for all accumulation steps in that global batch. Note that the value
   * of *N* is captured by AnchorReturnType.
   *
   * The buffer shape required for this anchor in IStepIO is [`batchesPerStep /
   * N`, `accumulationFactor`, `replicationFactor`, `<anchorTensorShape>`] (with
   * dimensions of size 1 removed).
   */
  EveryN,
  /**
   * Return the tensor value for *all* micro batches for each replica.
   *
   * The buffer shape required for this anchor in IStepIO is [`batchesPerStep`,
   * `accumulationFactor`, `replicationFactor`, `<anchorTensorShape>`] (with
   * dimensions of size 1 removed).
   */
  All,
  /**
   * Return one tensor value for each replica, doing a sum reduction over the
   * `batchesPerStep` and `accumulationFactor` dimensions.
   *
   * The buffer shape required for this anchor in IStepIO is
   * [`replicationFactor`, `<anchorTensorShape>`] (with dimensions of size 1
   * removed).
   */
  Sum,
};

std::ostream &operator<<(std::ostream &, AnchorReturnTypeId);

/**
 * Class that captures an \ref AnchorReturnTypeId value.
 *
 * When the value is \c AnchorReturnTypeId::EVERYN, the associated *N* value.
 * The constructor takes `std::string` values and converts them as appropriate.
 */
class AnchorReturnType {
public:
  /// Default constructor for the AnchorReturnType class.
  AnchorReturnType();

  /**
   * Constructor for the AnchorReturnType class.
   *
   * **NOTE**: Attempting to construct an AnchorReturnType for \c
   * AnchorReturnTypeId::EVERYN using this constructor will result in an error.
   * Use AnchorReturnType(std::string,int,TileSet,ExchangeStrategy) which also
   * specifies the return period.
   *
   * \param artString The string to convert to an AnchorReturnTypeId value.
   *     The following values are acceptable (case insensitive):
   *       * "final" = \c AnchorReturnTypeId::FINAL
   *       * "all" = \c AnchorReturnTypeId::ALL
   *       * "sum" = \c AnchorReturnTypeId::SUM
   * \param tileSet (Optional) The type of the tile set. Default:
   *     TileSet::Compute.
   * \param exchangeStrategy (Optional) The overlap strategy (between IO and
   *     compute) for anchor tensors. Default: ExchangeStrategy::JustInTime.
   */
  AnchorReturnType(
      std::string artString,
      TileSet tileSet                   = TileSet::Compute,
      ExchangeStrategy exchangeStrategy = ExchangeStrategy::JustInTime);

  /**
   * Constructor for the AnchorReturnType class.
   *
   * \param artString The string to convert to an AnchorReturnTypeId value.
   *     The following values are acceptable (case insensitive):
   *       * "final" = \c AnchorReturnTypeId::FINAL
   *       * "all" = \c AnchorReturnTypeId::ALL
   *       * "sum" = \c AnchorReturnTypeId::SUM
   * \param returnPeriod The value of *N* in the case of \c
   *     AnchorReturnTypeId::EVERYN.
   * \param tileSet (Optional) The type of the tile set. Default:
   *     TileSet::Compute.
   * \param exchangeStrategy (Optional) The overlap strategy (between IO and
   *     compute) for anchor tensors. Default: ExchangeStrategy::JustInTime.
   */
  AnchorReturnType(
      std::string artString,
      int returnPeriod,
      TileSet tileSet                   = TileSet::Compute,
      ExchangeStrategy exchangeStrategy = ExchangeStrategy::JustInTime);

  // Get the AnchorReturnTypeId. Not part of the public API.
  AnchorReturnTypeId id() const { return artId_; }

  // Get the associated return period (*N*). Not part of public API.
  // This applies when AnchorReturnTypeId is \c AnchorReturnTypeId::EVERYN.
  int rp() const;

  // Get a hash value. Not part of public API.
  std::size_t hash() const;

  /// Get a string of AnchorReturnTypeId.
  const std::string &str() const { return artStr_; }

  /// Get the type of the tile set.
  const TileSet &tileSet() const { return tileSet_; }

  /// Get the type of overlap strategy.
  const ExchangeStrategy &exchangeStrategy() const { return exchangeStrategy_; }

private:
  AnchorReturnTypeId getIdFromStr(std::string artString);

  std::string artStr_;
  AnchorReturnTypeId artId_;

  int returnPeriod_;

  TileSet tileSet_;
  ExchangeStrategy exchangeStrategy_;
};

using AnchorReturnTypeMap = std::map<TensorId, AnchorReturnType>;

/**
 * Class that describes the TileSet, ExchangeStrategy, and
 * ReplicatedStreamMode used for an input tensor.
 */
class InputSettings {
public:
  /// Constructor for the InputSettings class.
  InputSettings();

  /**
   * Constructor for the InputSettings class.
   *
   * \param tileSet The type of the tile set.
   * \param exchangeStrategy The overlap strategy (between IO and
   *     compute) for anchor tensors.
   */
  InputSettings(TileSet tileSet, ExchangeStrategy exchangeStrategy);

  /**
   * Constructor for the InputSettings class.
   *
   * \param replicatedStreamMode The mode used for the replicated stream.
   */
  InputSettings(ReplicatedStreamMode replicatedStreamMode);

  /// Get the type of the tile set.
  const TileSet &tileSet() const { return tileSet_; }

  /// Get the type of overlap strategy.
  const ExchangeStrategy &exchangeStrategy() const { return exchangeStrategy_; }

  /// Get the mode of the replicated stream.
  ReplicatedStreamMode replicatedStreamMode() const {
    return replicatedStreamMode_;
  }

  /**
   * Set the type of the tile set.
   *
   * \param tileSet The type of the tile set..
   */
  void setTileSet(TileSet tileSet) { tileSet_ = tileSet; }

  /**
   * Set the overlap strategy (between IO and compute).
   *
   * \param exchangeStrategy The overlap strategy.
   */
  void setExchangeStrategy(ExchangeStrategy exchangeStrategy) {
    exchangeStrategy_ = exchangeStrategy;
  }

  /**
   * Set the mode used for the replicated stream.
   *
   * \param replicatedStreamMode The mode used for the replicated stream.
   */
  void setReplicatedStreamMode(ReplicatedStreamMode streamMode) {
    replicatedStreamMode_ = streamMode;
  }

private:
  TileSet tileSet_;
  ExchangeStrategy exchangeStrategy_;
  ReplicatedStreamMode replicatedStreamMode_;
};

std::ostream &operator<<(std::ostream &, InputSettings);

/**
 * This class specifies parameters for host-device data streams.
 *
 * The parameters are used to control the amount input data processed in each
 * step, that is each Session::run call. The parameters also determine how data
 * is returned to the user.
 *
 * \sa AnchorReturnType, AnchorReturnTypeId.
 */
class DataFlow {
public:
  /**
   * Default constructor.
   *
   * This constructor sets `batchesPerStep` to 0 and does not have any anchor
   * tensors.
   */
  DataFlow();
  /**
   * Construct a DataFlow instance without anchor tensors.
   *
   * \param batchesPerStep The number of global batches to run in the inference
   *     or training session for each call to Session::run before returning
   *     control to the caller.
   */
  DataFlow(int batchesPerStep);
  /**
   * Construct a DataFlow instance with anchor tensors.
   *
   * \param batchesPerStep The number of global batches to run in the inference
   *     or training session for each call to Session::run before returning
   *     control to the caller.
   * \param anchorMap A mapping from output tensor TensorId to AnchorReturnType
   *     indicating the strategy with which to write the anchor tensor values
   *     to the IStepIO object provided to Session::run.
   */
  DataFlow(int batchesPerStep, const AnchorReturnTypeMap &anchorMap);
  /**
   * Construct a DataFlow instance with anchor tensors.
   *
   * \param batchesPerStep The number of global batches to run in the inference
   *     or training session for each call to Session::run before returning
   *     control to the caller.
   * \param anchorTensorIds The tensor ID of anchor tensors.
   * \param anchorReturnType The strategy with which to write anchor tensor
   *     values to the IStepIO object provided to Session::run.
   */
  DataFlow(int batchesPerStep,
           const std::vector<TensorId> anchorTensorIds,
           const AnchorReturnType &anchorReturnType = AnchorReturnType("All"));

  // Default copy constructor, not currently part of public API.
  DataFlow(const DataFlow &rhs) = default;
  // Default assignment operator, not currently part of public API.
  DataFlow &operator=(const DataFlow &rhs) = default;

  // Determine if a tensor is an anchor, not currently part of public API.
  bool isAnchored(TensorId) const;
  // Determine if batch counting is required, not currently part of public API.
  bool isBatchCountingRequired() const;
  // Return a vector of all anchor tensors, not currently part of public API.
  const std::vector<TensorId> &anchors() const { return v_anchors; }
  // Return a vector of all return periods, not currently part of public API.
  const std::vector<int> &rps() const { return v_rps; }
  // Number of anchor tensors, not currently part of public API.
  int nAnchors() const { return static_cast<int>(v_anchors.size()); }
  // Number of global batches per Session::run call, not currently part of
  // public API.
  int batchesPerStep() const { return batchesPerStep_; }
  // Get AnchorReturnType for a anchor tensor, not currently part of public API.
  AnchorReturnType art(TensorId anchorId) const;
  // Get number of fetches per replica for a tensor, not currently part of
  // public API.
  unsigned numOutFetchesPerRepl(const struct SessionOptions &opts,
                                const TensorId &anchorId) const;
  // Get a hash for this object, not currently part of public API.
  std::size_t hash() const;
  // Get the map of anchor return type, not currently part of public API.
  const AnchorReturnTypeMap &getAnchorReturnTypeMap() const {
    return m_anchors;
  }
  /// Set the value for `batchesPerStep`.
  void setBatchesPerStep(const int batchesPerStep) {
    batchesPerStep_ = batchesPerStep;
  }

private:
  // The number of batches processed by the backend in one call to train
  // or infer.
  // For an \c InferenceSession this is equal to the number of executions of
  // the model.
  // For a \c TrainingSession this is equal to the number of weight updates.
  int batchesPerStep_;

  // The set of tensors to return to the user after execution, and how
  // frequently they are returned during multi-batch training or inference
  AnchorReturnTypeMap m_anchors;

  // The set of anchor tensors (as std::vector).
  std::vector<TensorId> v_anchors;
  // The set of anchor tensors (as std::set).
  std::set<TensorId> s_anchors;

  // The unique set of return periods for all anchors.
  // Depending on the anchor return type, extra tensors are added
  // to the graph during its construction to keep track of batch
  // count. This member ensures the minimum number of tensors are
  // added.
  std::vector<int> v_rps;

  void isValidAnchorReturnPeriod(TensorId anchorId, int batchesPerStep);
};

} // namespace popart

namespace std {
template <> struct hash<popart::DataFlow> {
  std::size_t operator()(const popart::DataFlow &df) const { return df.hash(); }
};

template <> struct hash<popart::AnchorReturnType> {
  std::size_t operator()(const popart::AnchorReturnType &art) const {
    return art.hash();
  }
};
} // namespace std

#endif
